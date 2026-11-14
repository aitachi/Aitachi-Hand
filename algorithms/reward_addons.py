"""Reward-side add-ons evaluated on the PPO backbone.

- PBRSWrapper: potential-based reward shaping F = gamma * Phi(s') - Phi(s)
  (Ng et al., 1999). Phi is the negative mean normalized pose error, so the
  shaping is policy-invariant and telescopes over the episode. Time-limit
  truncation receives no terminal-potential correction (documented deviation;
  success terminates at Phi ~ 0 so the bias is limited to timeout episodes).

- RND: Random Network Distillation (Burda et al., 2018), practical
  single-value-head simplification: intrinsic reward = normalized prediction
  error of a predictor MLP regressing a fixed random target MLP on the
  observation, added to the external reward with weight lambda. The
  two-value-head / non-episodic return of the paper is NOT reimplemented;
  the report states this deviation explicitly.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn


class PBRSWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, potential_scale: float = 2.0, gamma: float = 0.99):
        super().__init__(env)
        self.potential_scale = float(potential_scale)
        self.gamma = float(gamma)
        self._last_potential = 0.0
        self.shaping_stats = {"mean_abs_shaping": 0.0, "n": 0}

    def _potential(self, obs):
        # obs layout: [q_norm(16), vel/10(16), prev_a(16), target_norm(16), err_norm(16), (phase...)]
        base = self.env.n_joints if hasattr(self.env, "n_joints") else (len(obs) - 4) // 5
        err_norm = obs[4 * base:5 * base]
        return -self.potential_scale * float(np.mean(np.abs(err_norm)))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_potential = self._potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_potential = self._potential(obs)
        shaping = self.gamma * new_potential - self._last_potential
        self._last_potential = new_potential
        self.shaping_stats["mean_abs_shaping"] = (
            (self.shaping_stats["mean_abs_shaping"] * self.shaping_stats["n"] + abs(shaping))
            / (self.shaping_stats["n"] + 1))
        self.shaping_stats["n"] += 1
        info["pbrs_shaping"] = float(shaping)
        return obs, reward + shaping, terminated, truncated, info


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


class RNDModule:
    """Fixed random target network + trained predictor; call .update() per rollout."""

    def __init__(self, obs_dim: int, hidden: int = 256, emb_dim: int = 64,
                 lr: float = 1e-4, device: str = "auto", intrinsic_weight: float = 0.5):
        self.device = th.device(device if device != "auto" and th.cuda.is_available() else "cpu")
        self.target = _MLP(obs_dim, hidden, emb_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = _MLP(obs_dim, hidden, emb_dim).to(self.device)
        self.optimizer = th.optim.Adam(self.predictor.parameters(), lr=lr)
        self.intrinsic_weight = float(intrinsic_weight)
        self._err_std_running = 1.0
        self.intrinsic_history: list[float] = []

    @th.no_grad()
    def intrinsic_reward(self, obs: np.ndarray) -> float:
        x = th.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        err = th.mean((self.target(x) - self.predictor(x)) ** 2, dim=-1)
        raw = float(err.mean().item())
        self._err_std_running = 0.999 * self._err_std_running + 0.001 * max(raw, 1e-8)
        return self.intrinsic_weight * raw / (self._err_std_running + 1e-8)

    def update(self, obs_batch: np.ndarray, epochs: int = 2, batch_size: int = 512):
        x = th.as_tensor(np.asarray(obs_batch, dtype=np.float32), device=self.device)
        n = x.shape[0]
        if n < batch_size:
            batch_size = max(16, n)
        for _ in range(epochs):
            perm = th.randperm(n, device=self.device)
            for start in range(0, n - batch_size + 1, batch_size):
                idx = perm[start:start + batch_size]
                pred = self.predictor(x[idx])
                with th.no_grad():
                    tgt = self.target(x[idx])
                loss = th.mean((pred - tgt) ** 2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.intrinsic_history.append(float(loss.item()))


class RNDRewardWrapper(gym.Wrapper):
    """Adds RND intrinsic reward at step time using the current predictor."""

    def __init__(self, env: gym.Env, module: RNDModule):
        super().__init__(env)
        self.module = module

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        intrinsic = self.module.intrinsic_reward(obs)
        info["rnd_intrinsic"] = float(intrinsic)
        return obs, reward + intrinsic, terminated, truncated, info
