"""Baseline builders: PPO / TD3 / SAC (Stable-Baselines3 2.x).

Hyperparameters follow the algorithm papers' continuous-control settings and
the validated v3 PPO recipe (AHand/style1101, experiments/ppo_150k_v3).
"""
from __future__ import annotations

import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3.common.noise import NormalActionNoise

PPO_HPARAMS = dict(
    learning_rate=3e-4,
    n_steps=256,
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=5e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_net=[256, 256],
    log_std_init=-1.5,
)

TD3_HPARAMS = dict(
    learning_rate=1e-3,
    buffer_size=300_000,
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "step"),
    gradient_steps=1,
    action_noise=None,  # set per-task from action dim
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    policy_net=[400, 300],
)

SAC_HPARAMS = dict(
    learning_rate=3e-4,
    buffer_size=300_000,
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "step"),
    gradient_steps=1,
    policy_net=[256, 256],
)


def make_ppo(env, seed: int, hparams=None, device="auto"):
    hp = {**PPO_HPARAMS, **(hparams or {})}
    return sb3.PPO(
        "MlpPolicy", env,
        learning_rate=hp["learning_rate"], n_steps=hp["n_steps"],
        batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
        gamma=hp["gamma"], gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef"], vf_coef=hp["vf_coef"], max_grad_norm=hp["max_grad_norm"],
        policy_kwargs={"net_arch": hp["policy_net"], "log_std_init": float(hp["log_std_init"])},
        seed=seed, verbose=0, device=device,
    )


def make_td3(env, seed: int, hparams=None, device="auto"):
    hp = {**TD3_HPARAMS, **(hparams or {})}
    n_actions = env.action_space.shape[-1]
    hp["action_noise"] = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    return sb3.TD3(
        "MlpPolicy", env,
        learning_rate=hp["learning_rate"], buffer_size=hp["buffer_size"],
        learning_starts=hp["learning_starts"], batch_size=hp["batch_size"],
        tau=hp["tau"], gamma=hp["gamma"], train_freq=hp["train_freq"],
        gradient_steps=hp["gradient_steps"], action_noise=hp["action_noise"],
        policy_delay=hp["policy_delay"], target_policy_noise=hp["target_policy_noise"],
        target_noise_clip=hp["target_noise_clip"],
        policy_kwargs={"net_arch": {"pi": hp["policy_net"], "qf": hp["policy_net"]}},
        seed=seed, verbose=0, device=device,
    )


def make_sac(env, seed: int, hparams=None, device="auto"):
    hp = {**SAC_HPARAMS, **(hparams or {})}
    return sb3.SAC(
        "MlpPolicy", env,
        learning_rate=hp["learning_rate"], buffer_size=hp["buffer_size"],
        learning_starts=hp["learning_starts"], batch_size=hp["batch_size"],
        tau=hp["tau"], gamma=hp["gamma"], train_freq=hp["train_freq"],
        gradient_steps=hp["gradient_steps"],
        policy_kwargs={"net_arch": {"pi": hp["policy_net"], "qf": hp["policy_net"]}},
        seed=seed, verbose=0, device=device,
    )
