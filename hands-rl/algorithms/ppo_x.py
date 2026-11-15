"""PPO eXtended: one PPO backbone with switchable published improvements.

Variants (all trained on identical rollouts, seeds and network sizes unless
noted; deviations are documented in the experiment report):

- objective="clip" : standard PPO-Clip surrogate (Schulman et al., 2017).
- objective="rb"   : TR-PPO-RB rollback surrogate (Truly-PPO). Beyond the
  trust region the objective slope reverses, actively pulling the ratio back.
  The KL-triggered variant uses a per-sample |log ratio| proxy for the
  per-state KL (old-distribution parameters are not stored by SB3).
- objective="exo"  : ExO-PPO-style smooth compression: the log-ratio is softly
  saturated outside the trust region so the gradient decays smoothly instead
  of being hard-clipped to zero.
- espo=True        : ESPO early stopping across update epochs on average
  ratio drift (threshold 0.25) or approx KL (0.05).
- ppg=True         : Phasic Policy Gradient: shared-trunk network, fewer
  policy epochs, extra value-only epochs, and a periodic auxiliary phase that
  distills value features into the shared trunk under a KL anchor to the
  behaviour snapshot.
"""
from __future__ import annotations

import copy
import json

import gymnasium as gym
from collections import deque
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO


class PPOX(PPO):
    def __init__(self, *args, objective: str = "clip", espo: bool = False,
                 espo_drift: float = 0.25, espo_kl: float = 0.05,
                 rb_trigger_log_ratio: float = 0.5,
                 kl_target: float = 0.008,
                 ppg: bool = False, ppg_interval: int = 9,
                 ppg_aux_epochs: int = 6, ppg_kl_beta: float = 1.0,
                 ppg_policy_epochs: int = 3, ppg_value_epochs: int = 6,
                 ppg_buffer_rollouts: int = 3, **kwargs):
        if objective not in ("clip", "rb", "exo", "kl"):
            raise ValueError(objective)
        if ppg and kwargs.get("policy_kwargs", {}).get("net_arch") not in (None, [256, 256]):
            raise ValueError("PPG requires a shared trunk: net_arch=[256, 256]")
        if ppg and "policy_kwargs" not in kwargs:
            kwargs["policy_kwargs"] = {"net_arch": [256, 256], "log_std_init": -1.5}
        super().__init__(*args, **kwargs)
        self.objective = objective
        self.espo = bool(espo)
        self.espo_drift = float(espo_drift)
        self.espo_kl = float(espo_kl)
        self.rb_trigger_log_ratio = float(rb_trigger_log_ratio)
        self.kl_target = float(kl_target)
        self._kl_beta = 1.0
        self.ppg = bool(ppg)
        self.ppg_interval = int(ppg_interval)
        self.ppg_aux_epochs = int(ppg_aux_epochs)
        self.ppg_kl_beta = float(ppg_kl_beta)
        self.ppg_policy_epochs = int(ppg_policy_epochs)
        self.ppg_value_epochs = int(ppg_value_epochs)
        self.ppg_buffer_rollouts = int(ppg_buffer_rollouts)
        self._aux_obs_buffer = deque(maxlen=self.ppg_buffer_rollouts)
        self._rollout_counter = 0
        self.diagnostics: list[dict] = []

    # ------------------------------------------------------------------ utils
    def _surrogate(self, ratio, advantages, clip_range, approx_kl=None):
        if self.objective == "clip":
            return -th.min(advantages * ratio,
                           advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
        if self.objective == "kl":
            # PPO paper's adaptive-KL-penalty variant: rho*A - beta*KL(old||new),
            # beta adapted between epochs (PPO-1, Schulman et al. 2017, eq. 7-8).
            return -(ratio * advantages).mean() + self._kl_beta * approx_kl
        if self.objective == "rb":
            eps = clip_range
            c = (1 + eps) / (1 - eps)  # rollback coefficient from the Truly-PPO bound
            relu = F.relu
            # Rollback region: ratio out of trust bounds OR per-state KL proxy fires.
            trigger = th.abs(th.log(ratio.clamp_min(1e-8))) > self.rb_trigger_log_ratio
            pos, neg = advantages >= 0, advantages < 0
            out_high = th.relu(ratio - (1 + eps)) + trigger.float() * th.relu(ratio - 1.0)
            out_low = th.relu((1 - eps) - ratio) + trigger.float() * th.relu(1.0 - ratio)
            loss = advantages * ratio - c * (out_high * relu(advantages) + out_low * relu(-advantages))
            return -loss.mean()
        # objective == "exo": smoothly compress the log-ratio outside the region.
        eps = clip_range
        log_ratio = th.log(ratio.clamp_min(1e-8))
        soft = eps * th.tanh((log_ratio.clamp(-10, 10) - eps) / eps) + eps
        soft = th.where(log_ratio > eps, soft,
                        th.where(log_ratio < -eps, -eps * th.tanh((-log_ratio - eps) / eps) - eps, log_ratio))
        return -(advantages * th.exp(soft)).mean()

    # ------------------------------------------------------------------ train
    def train(self):
        all_stats = []
        entropy_losses, pg_losses, value_losses = [], [], []

        def epoch_loops(total_epochs, policy_epochs, value_only_epochs=0, record=True):
            nonlocal entropy_losses, pg_losses, value_losses
            stats = {"ratio_mean": 0.0, "ratio_max": 0.0, "clip_frac": 0.0,
                     "approx_kl": 0.0, "drift": 0.0, "n_minibatch": 0, "stopped_epoch": None}
            stop = False
            for epoch in range(total_epochs):
                if stop:
                    stats["stopped_epoch"] = epoch
                    break
                eps = float(self.clip_range(1.0))
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions.to(self.device)
                    if isinstance(self.action_space, gym.spaces.Box):
                        actions = th.clamp(actions, -1, 1)
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, actions)
                    advantages = rollout_data.advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    with th.no_grad():
                        old_log_prob = rollout_data.old_log_prob
                        log_ratio = log_prob - old_log_prob
                        ratio = th.exp(log_ratio)
                        approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio)
                        clip_frac = th.mean((th.abs(ratio - 1) > eps).float())

                    value_only = bool(value_only_epochs) and epoch >= policy_epochs
                    if value_only:
                        policy_loss = th.zeros((), device=self.device)
                        entropy_loss = th.zeros((), device=self.device)
                    else:
                        policy_loss = self._surrogate(ratio, advantages, eps, approx_kl=approx_kl)
                        entropy_loss = -th.mean(entropy)
                    value_loss = F.mse_loss(rollout_data.returns.flatten(), values.flatten())
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                    pg_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    if not value_only:
                        entropy_losses.append(entropy_loss.item())
                    b = stats["n_minibatch"]
                    stats["ratio_mean"] = (stats["ratio_mean"] * b + ratio.mean().item()) / (b + 1)
                    stats["ratio_max"] = max(stats["ratio_max"], ratio.max().item())
                    stats["clip_frac"] = (stats["clip_frac"] * b + clip_frac.item()) / (b + 1)
                    stats["approx_kl"] = (stats["approx_kl"] * b + approx_kl.item()) / (b + 1)
                    stats["n_minibatch"] = b + 1
                stats["drift"] = abs(stats["ratio_mean"] - 1.0)
                if self.objective == "kl":
                    # Adaptive penalty coefficient between epochs (PPO paper eq. 8).
                    if stats["approx_kl"] > 1.5 * self.kl_target:
                        self._kl_beta *= 2.0
                    elif stats["approx_kl"] < self.kl_target / 1.5:
                        self._kl_beta = max(1e-6, self._kl_beta / 2.0)
                if self.espo and (stats["drift"] > self.espo_drift or stats["approx_kl"] > self.espo_kl):
                    stop = True
                    stats["stopped_epoch"] = epoch + 1
            stats["kl_beta"] = self._kl_beta
            return stats

        n_policy_epochs = self.ppg_policy_epochs if self.ppg else self.n_epochs
        value_extra = self.ppg_value_epochs if self.ppg else 0
        stats = epoch_loops(n_policy_epochs + value_extra, n_policy_epochs,
                            value_only_epochs=value_extra)
        stats.update(
            entropy_loss=float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            policy_loss=float(np.mean(pg_losses)),
            value_loss=float(np.mean(value_losses)),
        )
        all_stats.append(stats)

        # PPG auxiliary phase: distil value into the shared trunk under a KL anchor.
        if self.ppg:
            self._aux_obs_buffer.append((
                self.rollout_buffer.observations.copy(),
                self.rollout_buffer.returns.copy(),
                self.rollout_buffer.actions.copy(),
            ))
            self._rollout_counter += 1
            if self._rollout_counter % self.ppg_interval == 0:
                obs_all = np.concatenate([o for o, _, _ in self._aux_obs_buffer], axis=0)
                ret_all = np.concatenate([r for _, r, _ in self._aux_obs_buffer], axis=0)
                act_all = np.concatenate([a for _, _, a in self._aux_obs_buffer], axis=0)
                # Snapshot semantics WITHOUT deepcopy (torch forbids deepcopying
                # modules holding non-leaf tensors): the KL anchor target is the
                # policy AT AUX START, so precompute its log-probs once, before
                # any auxiliary update touches the parameters.
                with th.no_grad():
                    obs_t = th.as_tensor(obs_all, dtype=th.float32, device=self.device)
                    act_t = th.as_tensor(np.clip(act_all, -1, 1), dtype=th.float32, device=self.device)
                    old_log_prob_all = self.policy.get_distribution(obs_t).log_prob(act_t)
                aux_kl_losses, aux_value_losses = [], []
                for _ in range(self.ppg_aux_epochs):
                    perm = th.randperm(obs_all.shape[0], device=self.device)
                    for start in range(0, len(perm) - self.batch_size + 1, self.batch_size):
                        idx = perm[start:start + self.batch_size]
                        obs = th.as_tensor(obs_all, dtype=th.float32, device=self.device)[idx]
                        ret = th.as_tensor(ret_all, dtype=th.float32, device=self.device)[idx]
                        act = th.as_tensor(np.clip(act_all, -1, 1), dtype=th.float32, device=self.device)[idx]
                        values = self.policy.predict_values(obs)
                        value_loss = F.mse_loss(ret.flatten(), values.flatten())
                        new_log_prob = self.policy.get_distribution(obs).log_prob(act)
                        kl_loss = th.mean(old_log_prob_all[idx] - new_log_prob)
                        loss = value_loss + self.ppg_kl_beta * kl_loss
                        self.policy.optimizer.zero_grad()
                        loss.backward()
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.policy.optimizer.step()
                        aux_value_losses.append(value_loss.item())
                        aux_kl_losses.append(kl_loss.item())
                stats["aux_value_loss"] = float(np.mean(aux_value_losses))
                stats["aux_kl"] = float(np.mean(aux_kl_losses))

        self._n_updates += self.n_epochs
        from stable_baselines3.common.utils import explained_variance
        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())
        entry = {
            "updates": self._n_updates,
            "timesteps": self.num_timesteps,
            **{k: float(v) if v is not None else None for k, v in stats.items()},
            "explained_variance": float(explained_var),
        }
        self.diagnostics.append(entry)
        self.logger.record("train/entropy_loss", np.mean(entropy_losses) if entropy_losses else 0.0)
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", stats["approx_kl"])
        self.logger.record("train/clip_frac", stats["clip_frac"])

    def export_diagnostics(self, path):
        Path(path).write_text(json.dumps(self.diagnostics, indent=1), encoding="utf-8")


def make_ppox(env, seed: int, variant: str, device="auto", hparams=None):
    from .baselines import PPO_HPARAMS
    hp = {**PPO_HPARAMS, **(hparams or {})}
    shared = {
        "clip": dict(objective="clip"),
        "rb": dict(objective="rb"),
        "exo": dict(objective="exo"),
        "kl": dict(objective="kl"),
        "espo": dict(objective="clip", espo=True),
        "ppg": dict(objective="clip", ppg=True),
    }[variant]
    net_arch = [256, 256] if shared.get("ppg") else hp["policy_net"]
    return PPOX(
        "MlpPolicy", env,
        learning_rate=hp["learning_rate"], n_steps=hp["n_steps"],
        batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
        gamma=hp["gamma"], gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef"], vf_coef=hp["vf_coef"], max_grad_norm=hp["max_grad_norm"],
        policy_kwargs={"net_arch": net_arch, "log_std_init": float(hp["log_std_init"])},
        seed=seed, verbose=0, device=device, **shared,
    )
