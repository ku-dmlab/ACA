from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Optional

import jax, jax.numpy as jnp
import haiku as hk
import optax
import math

from relax.network.blocks import Activation, TimestepDiffQNet
from relax.utils.jax_utils import random_key_from_data
from relax.utils.diffusion import BetaScheduleCoefficients

@dataclass(frozen=True)
class ACADiffusion:
    num_timesteps: int
    q_grad_norm: bool
    guidance_weight: float = 1.0
    beta_schedule_type: str = 'linear'
    
    def beta_schedule(self):
        with jax.ensure_compile_time_eval():
            if self.beta_schedule_type == 'linear':
                betas = BetaScheduleCoefficients.linear_beta_schedule(self.num_timesteps)
            elif self.beta_schedule_type == 'cosine':
                betas = BetaScheduleCoefficients.cosine_beta_schedule(self.num_timesteps)
            return BetaScheduleCoefficients.from_beta(betas)

    def p_mean_variance(self, t: int, x: jax.Array, noise_pred: jax.Array):
        B = self.beta_schedule()
        x_recon = x * B.sqrt_recip_alphas_cumprod[t] - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
        x_recon = jnp.clip(x_recon, -1, 1)
        model_mean = x_recon * B.posterior_mean_coef1[t] + x * B.posterior_mean_coef2[t]
        model_log_variance = B.posterior_log_variance_clipped[t]
        return model_mean, model_log_variance
    
    def get_recon(self, t: int, x: jax.Array, noise: jax.Array):
        B = self.beta_schedule()
        x_recon = x * B.sqrt_recip_alphas_cumprod[t][:, jnp.newaxis] - noise * B.sqrt_recipm1_alphas_cumprod[t][:, jnp.newaxis]
        return x_recon

    def q_sample(self, t: int, x_start: jax.Array, noise: jax.Array):
        B = self.beta_schedule()
        return B.sqrt_alphas_cumprod[t] * x_start + B.sqrt_one_minus_alphas_cumprod[t] * noise
    
    def get_std(self, t: int):
        B = self.beta_schedule()
        return jnp.exp(0.5 * B.posterior_log_variance_clipped[t])
    
    def sample(self, key: jax.Array, model, shape: Tuple[int, ...]) -> jax.Array:
        x_key, noise_key = jax.random.split(key)
        x = jax.random.normal(x_key, shape)
        noise = jax.random.normal(noise_key, (self.num_timesteps, *shape))

        def body_fn(x, input):
            t, noise = input
            def grad_fn(x):
                q1, q2 = model(x, t)
                return q1.sum() + q2.sum()

            grad_x = jax.grad(grad_fn)(x)
            if self.q_grad_norm:
                grad_x = grad_x / (jnp.linalg.norm(grad_x, axis=-1, keepdims=True) + 1e-8)
            std = self.get_std(t)
            grad_x = -self.guidance_weight * std * grad_x

            model_mean, model_log_variance = self.p_mean_variance(t, x, grad_x)
            x = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * noise
            return x, None
        
        t = jnp.arange(self.num_timesteps)[::-1]
        x, _ = jax.lax.scan(body_fn, x, (t, noise))
        x = jnp.clip(x, -1.0, 1.0)
        return x
    

class ACAParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    log_alpha: jax.Array
    
@dataclass
class ACANet:
    q: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    
    num_timesteps: int
    q_grad_norm: bool
    guidance_weight: float
    act_dim: int
    beta_schedule_type: str = 'cosine'
    num_particles: int = 1
    target_entropy: float = 0.0

    @property
    def diffusion(self) -> ACADiffusion:
        return ACADiffusion(self.num_timesteps, self.q_grad_norm, self.guidance_weight, self.beta_schedule_type)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """for data collection"""

        q1_params, q2_params, log_alpha = policy_params

        original_obs_ndim = obs.ndim
        obs = obs[None, :] if obs.ndim == 1 else obs  

        def model_fn(x, t):
            q1 = self.q(q1_params, obs, x, t)
            q2 = self.q(q2_params, obs, x, t)
            return q1, q2    
        
        def q1s_for_actions(actions):
            return jax.vmap(self.q, in_axes=(None, 0, 0, 0))(q1_params, obs, actions, t_zero)
        def q2s_for_actions(actions):
            return jax.vmap(self.q, in_axes=(None, 0, 0, 0))(q2_params, obs, actions, t_zero)
        
        key, noise_key = jax.random.split(key)
        if self.num_particles == 1:
            act = self.diffusion.sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
        else:
            keys = jax.random.split(key, self.num_particles)
            acts = jax.vmap(self.diffusion.sample, in_axes=(0, None, None))(keys, model_fn, (*obs.shape[:-1], self.act_dim))
            
            t_zero = jnp.zeros((obs.shape[0],))
            q1s = jax.vmap(q1s_for_actions)(acts) 
            q2s = jax.vmap(q2s_for_actions)(acts) 
            qs = jnp.minimum(q1s, q2s)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)

            act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * 0.1 # 0.15 for humanoid-v4

        if original_obs_ndim == 1:
            act = act.squeeze(0)

        return act

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """for evaluation"""
        key = random_key_from_data(obs)
        q1_params, q2_params, log_alpha = policy_params
        log_alpha = -jnp.inf
        policy_params = (q1_params, q2_params, log_alpha)
        return self.get_action(key, policy_params, obs)


def create_aca_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 10,
    guidance_weight: float = 1.0,
    q_grad_norm: bool = True,
    beta_schedule_type: str = 'cosine',
    num_particles: int = 10,
    target_entropy_scale: float = 0.9, # 0.5 for humanoid-v4
) -> Tuple[ACANet, ACAParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act, t: TimestepDiffQNet(hidden_sizes, activation)(obs, act, t)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key = jax.random.split(key, 2)
        q1_params = q.init(q1_key, obs, act, 0)
        q2_params = q.init(q2_key, obs, act, 0)
        target_q1_params = q1_params
        target_q2_params = q2_params
        log_alpha = jnp.array(math.log(5), dtype=jnp.float32)
        return ACAParams(q1_params, q2_params, target_q1_params, target_q2_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = ACANet(q=q.apply, num_timesteps=num_timesteps, q_grad_norm=q_grad_norm, 
                    guidance_weight=guidance_weight, act_dim=act_dim, beta_schedule_type=beta_schedule_type, 
                    num_particles=num_particles, target_entropy=-act_dim*target_entropy_scale)
    return net, params