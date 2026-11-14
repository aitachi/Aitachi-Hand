"""Task 2: ahand_seq — fast gesture sequence OPEN -> V -> OK -> OPEN.

One episode chains three gesture phases; each phase must satisfy the v3
acceptance window (position + velocity + per-gesture tip geometry, held for
3 control cycles) before the target switches. Success = finishing all three
phases inside the total time budget. Per-phase arrival times are logged for
the phase-decomposition analysis.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .hand_core import GESTURES, HandCore
from .rewards import RewardWeights, compute_reward

PHASE_ORDER = ("V", "OK", "OPEN")


class AitachiSeqEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        phase_seconds: float = 2.0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        residual_scale: float = 0.3,
        max_command_rate_rad: float = 0.15,
        position_tolerance_rad: float = 0.12,
        velocity_tolerance_rad_s: float = 0.35,
        success_hold_steps: int = 3,
        ok_tip_gap_max_mm: float = 10.0,
        v_tip_separation_min_mm: float = 30.0,
        collision_failure_mm: float = 5.0,
        start_noise_std_rad: float = 0.02,
        phase_bonus: float = 8.0,
        final_bonus: float = 5.0,
        reward_weights=None,
        model_path=None,
        targets_path=None,
    ):
        super().__init__()
        self.core = HandCore(model_path=model_path, targets_path=targets_path,
                             control_dt=control_dt, physics_dt=physics_dt)
        self.n_joints = self.core.n_joints
        self.phase_steps = int(round(phase_seconds / control_dt))
        self.max_steps = self.phase_steps * len(PHASE_ORDER)
        self.max_episode_seconds = self.max_steps * control_dt
        self.residual_scale = float(residual_scale)
        self.max_command_rate_rad = float(max_command_rate_rad)
        self.position_tolerance_rad = float(position_tolerance_rad)
        self.velocity_tolerance_rad_s = float(velocity_tolerance_rad_s)
        self.success_hold_steps = int(success_hold_steps)
        self.ok_tip_gap_max_mm = float(ok_tip_gap_max_mm)
        self.v_tip_separation_min_mm = float(v_tip_separation_min_mm)
        self.collision_failure_mm = float(collision_failure_mm)
        self.start_noise_std_rad = float(start_noise_std_rad)
        self.phase_bonus_value = float(phase_bonus)
        self.final_bonus_value = float(final_bonus)
        self.reward_weights = reward_weights or RewardWeights()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.n_joints * 5 + 4,), dtype=np.float32)
        self.np_random = np.random.default_rng(1101)
        self.phase_index = 0
        self.phase_step = 0
        self.hold_count = 0
        self.phase_times: list[float] = []
        self.step_count = 0
        self.previous_mean_absolute_error = 0.0

    # -- helpers ----------------------------------------------------------------
    def _target(self):
        return self.core.targets[PHASE_ORDER[self.phase_index]]

    def _geometry_ok(self, m):
        gesture = PHASE_ORDER[self.phase_index]
        if gesture == "OK":
            return m["thumb_index_gap_mm"] <= self.ok_tip_gap_max_mm
        if gesture == "V":
            return m["index_middle_separation_mm"] >= self.v_tip_separation_min_mm
        return True

    def _observation(self):
        c = self.core
        q = c.data.qpos[c.qpos_ids]
        target = self._target()
        phase_onehot = np.zeros(len(PHASE_ORDER), dtype=np.float64)
        phase_onehot[self.phase_index] = 1.0
        phase_progress = np.array([self.phase_step / self.phase_steps], dtype=np.float64)
        return np.concatenate([
            c.normalized_position(q),
            c.data.qvel[c.qvel_ids] / 10.0,
            c.previous_action,
            c.normalized_position(target),
            c.normalized_error(q, target),
            phase_onehot,
            phase_progress,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.core.reset_state(self.core.targets["OPEN"], self.start_noise_std_rad, self.np_random)
        self.phase_index = 0
        self.phase_step = 0
        self.hold_count = 0
        self.phase_times = []
        self.step_count = 0
        err = self.core.normalized_error(self.core.data.qpos[self.core.qpos_ids], self._target())
        self.previous_mean_absolute_error = float(np.mean(np.abs(err)))
        info = {"is_success": False, "phase": PHASE_ORDER[0], "phase_times": []}
        info.update(self.core.metrics(self._target()))
        return self._observation(), info

    def step(self, action):
        target = self._target()
        action = self.core.apply_action(action, target, self.residual_scale, self.max_command_rate_rad)
        self.step_count += 1
        self.phase_step += 1
        finite = self.core.finite()
        if finite:
            m = self.core.metrics(target)
        else:
            m = {"max_position_error_rad": 1e6, "max_velocity_rad_s": 1e6, "max_normalized_force": 1e6,
                 "max_penetration_mm": 1e6, "thumb_index_gap_mm": 1e6, "index_middle_separation_mm": 1e6}
        collision_failure = m["max_penetration_mm"] > self.collision_failure_mm
        criteria_met = (m["max_position_error_rad"] <= self.position_tolerance_rad
                        and m["max_velocity_rad_s"] <= self.velocity_tolerance_rad_s
                        and self._geometry_ok(m))
        self.hold_count = self.hold_count + 1 if criteria_met and not collision_failure else 0
        phase_complete = self.hold_count >= self.success_hold_steps and finite and not collision_failure
        success = False
        phase_bonus = None
        if phase_complete:
            phase_time = (self.phase_step - self.success_hold_steps + 1) * self.core.control_dt
            self.phase_times.append(float(phase_time))
            remaining = max(0.0, 1.0 - self.phase_step / self.phase_steps)
            phase_bonus = self.phase_bonus_value + self.phase_bonus_value * remaining
            if self.phase_index == len(PHASE_ORDER) - 1:
                success = True
                phase_bonus += self.final_bonus_value
            else:
                self.phase_index += 1
                self.phase_step = 0
                self.hold_count = 0

        if finite:
            normalized_error = self.core.normalized_error(self.core.data.qpos[self.core.qpos_ids], self._target())
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
            self.step_count * self.core.control_dt, self.max_episode_seconds,
            success, self.reward_weights, within_tolerance=criteria_met,
            failure=collision_failure or not finite, phase_bonus=phase_bonus,
        )
        self.previous_mean_absolute_error = mean_error
        terminated = bool(success or not finite)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        total_time = sum(self.phase_times) if self.phase_times else None
        info = {"is_success": success, "phase": PHASE_ORDER[self.phase_index],
                "phases_completed": len(self.phase_times), "phase_times": list(self.phase_times),
                "sequence_time": total_time, "collision_failure": collision_failure,
                "reward_components": components, "elapsed_time": self.step_count * self.core.control_dt}
        info.update(m)
        return self._observation(), components["total"], terminated, truncated, info
