from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.aca import ACANet, ACAParams
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric


class ACAOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    log_alpha: optax.OptState

class ACATrainState(NamedTuple):
    params: ACAParams
    opt_state: ACAOptStates
    step: int

class ACA(Algorithm):
    def __init__(self, agent: ACANet, params: ACAParams, *, gamma: float = 0.99, lr: float = 3e-4,
                 tau: float = 0.005, reward_scale: float = 0.2, lr_schedule_end: float = 5e-5, delay_update: int = 2, delay_alpha_update: int = 250):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.optim = optax.adam(lr_schedule)
        self.reward_scale = reward_scale
        self.delay_update = delay_update
        self.alpha_optim = optax.adam(3e-2)
        self.delay_alpha_update = delay_alpha_update
        self.state = ACATrainState(
            params=params,
            opt_state=ACAOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: ACATrainState, data: Experience
        ) -> Tuple[ACATrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, log_alpha_opt_state = state.opt_state
            step = state.step
            batch_size = obs.shape[0]
            key, key1, key2, key3 = jax.random.split(key, 4)

            reward *= self.reward_scale

            # compute target q
            next_action = self.agent.get_action(key1, (q1_params, q2_params, log_alpha), next_obs)
            t_zero = jnp.zeros((batch_size,))
            q1_target = self.agent.q(target_q1_params, next_obs, next_action, t_zero)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action, t_zero)
            q_target = jnp.minimum(q1_target, q2_target)
            q_backup = reward + (1 - done) * self.gamma * q_target

            # compute q loss
            def q_td_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, obs, action, t_zero)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss

            q1_loss, q1_grads = jax.value_and_grad(q_td_loss_fn)(q1_params)
            q2_loss, q2_grads = jax.value_and_grad(q_td_loss_fn)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)

            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            # compute qt loss
            t = jax.random.randint(key2, (batch_size,), 0, self.agent.num_timesteps)
            a_T = jax.random.normal(key3, (batch_size, self.agent.act_dim))
            a_0 = action

            a_t = jax.vmap(self.agent.diffusion.q_sample)(t, a_0, a_T)

            q1 = self.agent.q(q1_params, obs, action, t_zero)
            q2 = self.agent.q(q2_params, obs, action, t_zero)
            q_cat = jnp.stack([q1, q2], axis=0)
            q_mean = jnp.mean(q_cat, axis=0)

            def q_t_loss_fn(q_params: hk.Params) -> jax.Array:
                q_t = self.agent.q(q_params, obs, a_t, t)
                q_loss = jnp.mean((q_t - q_mean) ** 2) 
                return q_loss

            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2)
                log_alpha_loss = -1 * log_alpha * (-1 * jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)
                return log_alpha_loss
            
            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda params, opt_state: param_update(optim, params, jax.grad(log_alpha_loss_fn)(params), opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            qt1_loss, qt1_grads = jax.value_and_grad(q_t_loss_fn)(q1_params)
            qt2_loss, qt2_grads = jax.value_and_grad(q_t_loss_fn)(q2_params)
            q1_params, q1_opt_state = delay_param_update(self.optim, q1_params, qt1_grads, q1_opt_state)
            q2_params, q2_opt_state = delay_param_update(self.optim, q2_params, qt2_grads, q2_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = optax.incremental_update(q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(q2_params, target_q2_params, self.tau)

            state = ACATrainState(
                params=ACAParams(q1_params, q2_params, target_q1_params, target_q2_params, log_alpha),
                opt_state=ACAOptStates(q1_opt_state, q2_opt_state, log_alpha_opt_state),
                step=step + 1,
            )
            info = {
                "td_q1_loss": q1_loss,
                "td_q2_loss": q2_loss,
                "q1": jnp.mean(q1),
                "q2": jnp.mean(q2),
                "qt1_loss": qt1_loss,
                "qt2_loss": qt2_loss,
                "log_alpha": log_alpha,
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)
    
    def get_policy_params(self):
        return self.state.params.q1, self.state.params.q2, self.state.params.log_alpha
