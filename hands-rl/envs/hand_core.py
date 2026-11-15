"""Shared MuJoCo ORCA-hand core for the two Aitachi-Hand RL tasks.

Reuses the numerically validated v3 recipe from AHand/style1101:
position actuators (kp=12, kv=0.5; wrist kp=60), implicitfast integrator,
0.002 s physics / 0.02 s control, residual absolute action mode with
0.15 rad/step command-rate limiting.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

GESTURES = ("V", "OK", "OPEN")
WRIST_JOINT = "right_wrist"

_MODEL_CANDIDATES = [
    Path(os.environ.get("AHAND_MODEL_ROOT", "")) / "scene_right.xml",
    Path("C:/Users/ASUS/Desktop/AHand/mujoco/models/orca_v2/scene_right.xml"),
    Path("/mnt/c/Users/ASUS/Desktop/AHand/mujoco/models/orca_v2/scene_right.xml"),
    Path(__file__).resolve().parents[2] / "assets/orca_v2/scene_right.xml",
]
_TARGETS_CANDIDATES = [
    Path(os.environ.get("AHAND_MODEL_ROOT", "")) / "../../output/gesture_targets.json",
    Path("C:/Users/ASUS/Desktop/AHand/mujoco/output/gesture_targets.json"),
    Path("/mnt/c/Users/ASUS/Desktop/AHand/mujoco/output/gesture_targets.json"),
    Path(__file__).resolve().parents[2] / "assets/gesture_targets.json",
]


def _first_existing(candidates):
    for path in candidates:
        if path and path.is_file():
            return str(path)
    raise FileNotFoundError(
        "ORCA v2 model not found. Set AHAND_MODEL_ROOT to the folder holding "
        "scene_right.xml (official repo: github.com/orcahand/orcahand_description), "
        "or run scripts/fetch_assets.py. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def default_model_path() -> str:
    return _first_existing(_MODEL_CANDIDATES)


def default_targets_path() -> str:
    for path in _TARGETS_CANDIDATES:
        if path and path.is_file():
            return str(path)
    return _first_existing(_TARGETS_CANDIDATES)


def load_targets(path: str) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = set(GESTURES) - raw.keys()
    if missing:
        raise ValueError(f"Missing gesture targets: {sorted(missing)}")
    return raw


class HandCore:
    """Loads the orca_v2 right-hand scene and exposes finger-joint plumbing."""

    def __init__(
        self,
        model_path: str | None = None,
        targets_path: str | None = None,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
    ):
        self.model_path = model_path or default_model_path()
        self.targets_path = targets_path or default_targets_path()
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.model.opt.timestep = physics_dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # Position-servo gains, identical to the validated gesture demo / v3 PPO setup.
        self.model.actuator_gainprm[:, 0] = 12.0
        self.model.actuator_biasprm[:, 1] = -12.0
        self.model.actuator_biasprm[:, 2] = -0.5
        wrist_actuator = self.model.actuator("right_wrist_actuator").id
        self.model.actuator_gainprm[wrist_actuator, 0] = 60.0
        self.model.actuator_biasprm[wrist_actuator, 1] = -60.0
        self.model.actuator_biasprm[wrist_actuator, 2] = -3.0
        self.data = mujoco.MjData(self.model)

        targets_by_gesture = load_targets(self.targets_path)
        self.joint_names = [n for n in targets_by_gesture["OPEN"] if n != WRIST_JOINT]
        self.n_joints = len(self.joint_names)
        self.joint_ids = np.array([self.model.joint(n).id for n in self.joint_names])
        self.qpos_ids = np.array([self.model.joint(n).qposadr[0] for n in self.joint_names])
        self.qvel_ids = np.array([self.model.joint(n).dofadr[0] for n in self.joint_names])
        self.actuator_ids = np.array([self.model.actuator(n + "_actuator").id for n in self.joint_names])
        self.wrist_qpos_id = self.model.joint(WRIST_JOINT).qposadr[0]
        self.wrist_qvel_id = self.model.joint(WRIST_JOINT).dofadr[0]
        self.wrist_actuator_id = wrist_actuator
        self.joint_lower = self.model.jnt_range[self.joint_ids, 0].copy()
        self.joint_upper = self.model.jnt_range[self.joint_ids, 1].copy()
        self.joint_span = self.joint_upper - self.joint_lower
        self.targets = {
            name: np.array([values[j] for j in self.joint_names], dtype=np.float64)
            for name, values in targets_by_gesture.items()
        }
        for name, target in self.targets.items():
            if np.any(target < self.joint_lower - 1e-8) or np.any(target > self.joint_upper + 1e-8):
                raise ValueError(f"Target {name} violates model joint limits")

        self.control_dt = float(control_dt)
        self.physics_dt = float(physics_dt)
        self.frame_skip = int(round(control_dt / physics_dt))

        # Tip sites for V/OK geometric acceptance (same bodies as the v3 env).
        self.tip_bodies = {
            "index": "right_I-FingerTipAssembly_ec49c16c",
            "middle": "right_M-FingerTipAssembly_34afb748",
            "thumb": "right_T-DP_b7429e50",
        }
        self.tip_points = {
            "index": np.array([-0.008, 0.0, 0.039]),
            "middle": np.array([0.0, 0.0, 0.042]),
            "thumb": np.array([-0.005, 0.0, 0.026]),
        }
        self.commanded_target = np.zeros(self.n_joints)
        self.previous_action = np.zeros(self.n_joints)

    # -- normalization helpers -------------------------------------------------
    def normalized_position(self, q):
        return 2.0 * (q - self.joint_lower) / self.joint_span - 1.0

    def normalized_error(self, q, target):
        return (q - target) / self.joint_span

    def denormalize(self, total):
        return self.joint_lower + (total + 1.0) * 0.5 * self.joint_span

    # -- physics ---------------------------------------------------------------
    def reset_state(self, initial_q, noise_std=0.0, rng=None):
        mujoco.mj_resetData(self.model, self.data)
        initial = np.array(initial_q, dtype=np.float64)
        if noise_std and rng is not None:
            initial = initial + rng.normal(0.0, noise_std, self.n_joints)
        initial = np.clip(initial, self.joint_lower, self.joint_upper)
        self.data.qpos[self.qpos_ids] = initial
        self.data.qvel[:] = 0.0
        self.data.qpos[self.wrist_qpos_id] = 0.0
        self.data.qvel[self.wrist_qvel_id] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.actuator_ids] = initial
        self.commanded_target = initial.copy()
        self.previous_action = np.zeros(self.n_joints)
        mujoco.mj_forward(self.model, self.data)
        return initial

    def apply_action(self, action, target, residual_scale=0.3, max_command_rate_rad=0.15):
        """Residual absolute action: zero policy output commands the target prior."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        prior = self.normalized_position(target)
        total = np.clip(prior + residual_scale * action, -1.0, 1.0)
        desired = self.denormalize(total)
        delta = np.clip(desired - self.commanded_target, -max_command_rate_rad, max_command_rate_rad)
        self.commanded_target = np.clip(self.commanded_target + delta, self.joint_lower, self.joint_upper)
        self.data.ctrl[self.actuator_ids] = self.commanded_target
        self.data.ctrl[self.wrist_actuator_id] = 0.0
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.previous_action = action.copy()
        return action

    def finite(self):
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    # -- metrics ---------------------------------------------------------------
    def tip_position(self, finger):
        body_id = self.model.body(self.tip_bodies[finger]).id
        return self.data.xpos[body_id] + self.data.xmat[body_id].reshape(3, 3) @ self.tip_points[finger]

    def tip_metrics_mm(self):
        index = self.tip_position("index")
        middle = self.tip_position("middle")
        thumb = self.tip_position("thumb")
        return {
            "thumb_index_gap_mm": float(np.linalg.norm(thumb - index) * 1000.0),
            "index_middle_separation_mm": float(np.linalg.norm(index - middle) * 1000.0),
        }

    def max_penetration_mm(self):
        return max((max(0.0, -c.dist) * 1000.0 for c in self.data.contact), default=0.0)

    def metrics(self, target):
        position = self.data.qpos[self.qpos_ids]
        velocity = self.data.qvel[self.qvel_ids]
        error = position - target
        force = self.data.actuator_force[self.actuator_ids]
        force_scale = np.max(np.abs(self.model.actuator_forcerange[self.actuator_ids]), axis=1)
        out = {
            "max_position_error_rad": float(np.max(np.abs(error))),
            "max_velocity_rad_s": float(np.max(np.abs(velocity))),
            "max_normalized_force": float(np.max(np.abs(force / force_scale))),
            "max_penetration_mm": self.max_penetration_mm(),
        }
        out.update(self.tip_metrics_mm())
        return out
