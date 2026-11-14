"""Reward components shared by both Aitachi-Hand tasks (v3-validated weights)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    position_weight: float = 4.0
    velocity_weight: float = 0.02
    velocity_free_rad_s: float = 0.30
    action_weight: float = 0.005
    force_weight: float = 0.01
    collision_weight: float = 2.0
    time_penalty: float = 0.02
    progress_gain: float = 2.0
    tolerance_bonus: float = 0.8
    success_bonus: float = 10.0
    success_time_bonus: float = 10.0
    failure_penalty: float = 2.0

    def to_dict(self):
        return asdict(self)


def compute_reward(
    normalized_error,
    velocity,
    action,
    previous_action,
    normalized_force,
    penetration_mm,
    previous_mean_absolute_error,
    elapsed_seconds,
    max_episode_seconds,
    success,
    weights: RewardWeights,
    within_tolerance=False,
    failure=False,
    phase_bonus=None,
):
    normalized_error = np.asarray(normalized_error, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    normalized_force = np.asarray(normalized_force, dtype=np.float64)
    mean_absolute_error = float(np.mean(np.abs(normalized_error)))
    collision_excess = max(0.0, float(penetration_mm) - 0.5) / 5.0
    velocity_excess = np.maximum(0.0, np.abs(velocity) - weights.velocity_free_rad_s)
    components = {
        "position": -weights.position_weight * float(np.mean(normalized_error**2)),
        "velocity": -weights.velocity_weight * float(np.mean(velocity_excess**2)),
        "action": -weights.action_weight * float(np.mean((action - previous_action) ** 2)),
        "force": -weights.force_weight * float(np.mean(normalized_force**2)),
        "collision": -weights.collision_weight * collision_excess,
        "time": -weights.time_penalty,
        "progress": weights.progress_gain * (previous_mean_absolute_error - mean_absolute_error),
        "tolerance": weights.tolerance_bonus if within_tolerance else 0.0,
        "success": 0.0,
        "failure": -weights.failure_penalty if failure else 0.0,
    }
    if phase_bonus is not None:
        components["phase"] = float(phase_bonus)
    if success:
        remaining_fraction = np.clip(1.0 - elapsed_seconds / max_episode_seconds, 0.0, 1.0)
        components["success"] = weights.success_bonus + weights.success_time_bonus * remaining_fraction
    components["total"] = float(sum(components.values()))
    return components, mean_absolute_error
