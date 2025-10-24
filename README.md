# Actor-Critic without Actor

This is an official repository of **"Actor-Critic without Actor"**.

<p align="center">
<font size=5>📑</font>
<a target="_self" href="https://arxiv.org/abs/2509.21022"> <img style="height:14pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a> 

## Installation

```bash
# Create environemnt
conda create -n aca python=3.11 numpy tqdm tensorboardX matplotlib scikit-learn black snakeviz ipykernel setproctitle numba pyyaml
conda activate aca

# One of: Install jax WITH CUDA 
pip install --upgrade "jax[cuda12]==0.4.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install package
pip install -r requirements.txt
pip install -e .
```

## Run
```bash
# Run one experiment
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python scripts/train_mujoco.py --alg aca
```

## Acknowledgement
We implemented our code based on [DACER](https://github.com/happy-yan/DACER-Diffusion-with-Online-RL) and [SDAC](https://github.com/mahaitongdae/diffusion_policy_online_rl),  and we greatly appreciate their promising works.

## Citation
```bash
@article{ki2025actor,
  title={Actor-Critic without Actor},
  author={Ki, Donghyeon and Ahn, Hee-Jun and Kim, Kyungyoon and Lee, Byung-Jun},
  journal={arXiv preprint arXiv:2509.21022},
  year={2025}
}
```

## Tips
1. Search "for humanoid-v4" in the code to find the parameters for Humanoid-v4 environments.
2. The version of MuJoCo is mujoco210.
