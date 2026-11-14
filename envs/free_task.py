"""Task 1: ahand_free — five fingers move freely to random reachable poses
without self-collision ("全面自由活动但不打架").

Each episode samples a target on the natural-pose manifold (Dirichlet mixture
of the three calibrated gesture landmarks + small noise). Success requires the
standard hold window with penetration <= 1 mm, i.e. collision-free arrival.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .hand_core import GESTURES, HandCore
from .rewards import RewardWeights, compute_reward


class AitachiFreeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        target_mix_concentration: float = 2.0,
        target_noise_std_rad: float = 0.05,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        max_episode_seconds: float = 2.5,
        residual_scale: float = 0.3,
        max_command_rate_rad: float = 0.15,
        position_tolerance_rad: float = 0.12,
        velocity_tolerance_rad_s: float = 0.35,
        success_hold_steps: int = 3,
        collision_failure_mm: float = 5.0,
        collision_free_mm: float = 1.0,
        start_noise_std_rad: float = 0.02,
        reward_weights=None,
        model_path=None,
        targets_path=None,
    ):
        super().__init__()
        self.core = HandCore(model_path=model_path, targets_path=targets_path,
                             control_dt=control_dt, physics_dt=physics_dt)
        self.n_joints = self.core.n_joints
        self.max_steps = int(round(max_episode_seconds / control_dt))
        self.target_mix_concentration = float(target_mix_concentration)
        self.target_noise_std_rad = float(target_noise_std_rad)
        self.residual_scale = float(residual_scale)
        self.max_command_rate_rad = float(max_command_rate_rad)
        self.position_tolerance_rad = float(position_tolerance_rad)
        self.velocity_tolerance_rad_s = float(velocity_tolerance_rad_s)
        self.success_hold_steps = int(success_hold_steps)
        self.collision_failure_mm = float(collision_failure_mm)
        self.collision_free_mm = float(collision_free_mm)
        self.start_noise_std_rad = float(start_noise_std_rad)
        self.reward_weights = reward_weights or RewardWeights()
        self.max_episode_seconds = float(max_episode_seconds)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.n_joints * 5,), dtype=np.float32)
        self.np_random = np.random.default_rng(1101)
        self.step_count = 0
        self.hold_count = 0
        self.episode_target = np.zeros(self.n_joints)
        self.previous_mean_absolute_error = 0.0

    def _sample_target(self):
        alpha = np.full(len(GESTURES), self.target_mix_concentration / len(GESTURES))
        weights = self.np_random.dirichlet(alpha)
        mixture = sum(w * self.core.targets[g] for g, w in zip(GESTURES, weights))
        mixture = mixture + self.np_random.normal(0.0, self.target_noise_std_rad, self.n_joints)
        return np.clip(mixture, self.core.joint_lower, self.core.joint_upper)

    def _observation(self):
        c = self.core
        q = c.data.qpos[c.qpos_ids]
        return np.concatenate([
            c.normalized_position(q),
            c.data.qvel[c.qvel_ids] / 10.0,
            c.previous_action,
            c.normalized_position(self.episode_target),
            c.normalized_error(q, self.episode_target),
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.episode_target = self._sample_target()
        self.core.reset_state(self.core.targets["OPEN"], self.start_noise_std_rad, self.np_random)
        self.step_count = 0
        self.hold_count = 0
        err = self.core.normalized_error(self.core.data.qpos[self.core.qpos_ids], self.episode_target)
        self.previous_mean_absolute_error = float(np.mean(np.abs(err)))
        metrics = self.core.metrics(self.episode_target)
        info = {"is_success": False, "arrival_time": None}
        info.update(metrics)
        return self._observation(), info

    def step(self, action):
        action = self.core.apply_action(action, self.episode_target,
                                        self.residual_scale, self.max_command_rate_rad)
        self.step_count += 1
        finite = self.core.finite()
        if finite:
            m = self.core.metrics(self.episode_target)
        else:
            m = {"max_position_error_rad": 1e6, "max_velocity_rad_s": 1e6,
                 "max_normalized_force": 1e6, "max_penetration_mm": 1e6,
                 "thumb_index_gap_mm": 1e6, "index_middle_separation_mm": 1e6}
        collision_failure = m["max_penetration_mm"] > self.collision_failure_mm
        collision_free = m["max_penetration_mm"] <= self.collision_free_mm
        criteria_met = (m["max_position_error_rad"] <= self.position_tolerance_rad
                        and m["max_velocity_rad_s"] <= self.velocity_tolerance_rad_s
                        and collision_free)
        self.hold_count = self.hold_count + 1 if criteria_met and not collision_failure else 0
        success = self.hold_count >= self.success_hold_steps and finite and not collision_failure
        arrival_time = (self.step_count - self.success_hold_steps + 1) * self.core.control_dt if success else None

        if finite:
            normalized_error = self.core.normalized_error(self.core.data.qpos[self.core.qpos_ids], self.episode_target)
            velocity = self.core.data.qvel[self.core.qvel_ids]
            force_scale = np.max(np.abs(self.core.model.actuator_forcerange[self.core.actuator_ids]), axis=1)
            normalized_force = self.core.data.actuator_force[self.core.actuator_ids] / force_scale
        else:
            normalized_error = np.full(self.n_joints, 10.0)
            velocity = np.full(self.n_joints, 10.0)
            normalized_force = np.full(self.n_joints, 10.0)
        components, mean_error = compute_reward(
            normalized_error, velocity, action, self.core.previous_action, normalized_force,
            m["max_penetration_mm"], self.previous_mean_absolute_error,
            arrival_time or self.step_count * self.core.control_dt, self.max_episode_seconds,
            success, self.reward_weights, within_tolerance=criteria_met,
            failure=collision_failure or not finite,
        )
        self.previous_mean_absolute_error = mean_error
        terminated = bool(success or not finite)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        info = {"is_success": success, "arrival_time": arrival_time,
                "collision_failure": collision_failure, "reward_components": components,
                "elapsed_time": self.step_count * self.core.control_dt}
        info.update(m)
        return self._observation(), components["total"], terminated, truncated, info
