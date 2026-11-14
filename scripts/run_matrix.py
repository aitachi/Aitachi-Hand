"""Single-run driver for the Aitachi-Hand comparison matrix.

Example:
  python scripts/run_matrix.py --task free --algo ppo_rb --seed 1 \
      --timesteps 150000 --out results
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms import (PBRSWrapper, RNDRewardWrapper, RNDModule,  # noqa: E402
                        make_ppo, make_ppox, make_sac, make_td3, make_trpo)
from envs import TASKS  # noqa: E402

N_ENVS_ONPOLICY = 8
EVAL_SEEDS = [7001 + i for i in range(25)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EpisodeLogger(BaseCallback):
    def __init__(self, out_path: Path, rnd: RNDModule | None = None):
        super().__init__()
        self.out_path = out_path
        self.rows = []
        self.rnd = rnd

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self.rows.append({
                "timesteps": int(self.num_timesteps),
                "reward": float(info["episode"]["r"]),
                "length": int(info["episode"]["l"]),
                "is_success": bool(info.get("is_success", False)),
                "arrival_time": info.get("arrival_time"),
                "sequence_time": info.get("sequence_time"),
                "phase_times": info.get("phase_times"),
                "collision_failure": bool(info.get("collision_failure", False)),
                "max_penetration_mm": info.get("max_penetration_mm"),
            })
        return True

    def _on_rollout_end(self):
        if self.rnd is not None and hasattr(self.model, "rollout_buffer"):
            obs = self.model.rollout_buffer.observations
            flat = np.asarray(obs).reshape(-1, obs.shape[-1])
            self.rnd.update(flat)

    def _on_training_end(self):
        with self.out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timesteps", "reward", "length", "is_success",
                                                  "arrival_time", "sequence_time", "phase_times",
                                                  "collision_failure", "max_penetration_mm"])
            writer.writeheader()
            for row in self.rows:
                out = dict(row)
                out["phase_times"] = json.dumps(out["phase_times"]) if out["phase_times"] is not None else ""
                out["is_success"] = int(out["is_success"])
                out["collision_failure"] = int(out["collision_failure"])
                writer.writerow(out)


def build_envs(task_id, n_envs, seed, reward_addon=None, rnd_module=None):
    entry = TASKS[task_id]

    def factory(rank):
        def create():
            import gymnasium as gym
            env = gym.make(entry)
            env.reset(seed=seed + 1000 * rank)
            if reward_addon == "pbrs":
                env = PBRSWrapper(env)
            elif reward_addon == "rnd":
                env = RNDRewardWrapper(env, rnd_module)
            return Monitor(env)
        return create

    return SubprocVecEnv([factory(i) for i in range(n_envs)], start_method="spawn")


def evaluate(model, task_id, episodes=25, deterministic=True):
    import gymnasium as gym
    from envs import TASKS
    rows = []
    for i, seed in enumerate(EVAL_SEEDS[:episodes]):
        env = Monitor(gym.make(TASKS[task_id]))
        obs, info = env.reset(seed=seed)
        done = False
        total_reward, steps, max_pen = 0.0, 0, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            max_pen = max(max_pen, float(info.get("max_penetration_mm", 0.0)))
            done = terminated or truncated
        rows.append({
            "seed": seed, "reward": total_reward, "steps": steps,
            "is_success": bool(info.get("is_success", False)),
            "arrival_time": info.get("arrival_time"),
            "sequence_time": info.get("sequence_time"),
            "phase_times": info.get("phase_times"),
            "max_penetration_mm": max_pen,
        })
        env.close()
    successes = [r for r in rows if r["is_success"]]
    finish_times = [r["arrival_time"] if r["arrival_time"] is not None
                    else r["sequence_time"] for r in successes]
    summary = {
        "episodes": len(rows),
        "success_rate": len(successes) / len(rows),
        "mean_finish_time": float(np.mean(finish_times)) if finish_times else None,
        "std_finish_time": float(np.std(finish_times)) if finish_times else None,
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_max_penetration_mm": float(np.mean([r["max_penetration_mm"] for r in rows])),
        "max_penetration_mm": float(np.max([r["max_penetration_mm"] for r in rows])),
    }
    if task_id == "seq" and successes:
        phase = np.array([r["phase_times"] for r in successes if r["phase_times"]], dtype=object)
        if len(phase):
            summary["mean_phase_times"] = [
                float(np.mean([p[i] for p in phase if len(p) == 3])) for i in range(3)
            ]
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--algo", required=True,
                        choices=["ppo", "td3", "sac", "trpo", "ppo_kl", "ppo_rb",
                                 "ppo_espo", "ppo_ppg", "ppo_exo", "ppo_pbrs", "ppo_rnd"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--out", default="results")
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out) / f"{args.task}_{args.algo}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "auto"

    rnd_module = None
    if args.algo == "ppo_rnd":
        probe = __import__("gymnasium", fromlist=["make"]).make(TASKS[args.task])
        obs_dim = probe.observation_space.shape[0]
        probe.close()
        rnd_module = RNDModule(obs_dim=int(obs_dim), device=device)
    reward_addon = {"ppo_pbrs": "pbrs", "ppo_rnd": "rnd"}.get(args.algo)

    onpolicy = args.algo.startswith("ppo")
    n_envs = N_ENVS_ONPOLICY if onpolicy else 1
    env = build_envs(args.task, n_envs, args.seed, reward_addon, rnd_module)

    if args.algo in ("ppo", "ppo_pbrs", "ppo_rnd"):
        model = make_ppo(env, args.seed, device=device)
    elif args.algo == "trpo":
        model = make_trpo(env, args.seed, device=device)
    elif args.algo == "td3":
        model = make_td3(env, args.seed, device=device)
    elif args.algo == "sac":
        model = make_sac(env, args.seed, device=device)
    else:
        variant = args.algo.replace("ppo_", "")
        model = make_ppox(env, args.seed, variant, device=device)

    t0 = time.time()
    logger = EpisodeLogger(out_dir / "episodes.csv", rnd=rnd_module)
    try:
        model.learn(total_timesteps=args.timesteps, callback=logger,
                    progress_bar=False)
    finally:
        env.close()
    wall_seconds = time.time() - t0

    model.save(out_dir / "final_model")
    if hasattr(model, "export_diagnostics"):
        model.export_diagnostics(out_dir / "ppo_diagnostics.json")

    eval_summary, eval_rows = evaluate(model, args.task, episodes=args.eval_episodes)
    with (out_dir / "eval_episodes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
        writer.writeheader()
        for row in eval_rows:
            row = dict(row)
            row["phase_times"] = json.dumps(row["phase_times"]) if row["phase_times"] is not None else ""
            row["is_success"] = int(row["is_success"])
            writer.writerow(row)
    (out_dir / "eval.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    from envs.hand_core import default_model_path, default_targets_path
    manifest = {
        "task": args.task, "algo": args.algo, "seed": args.seed,
        "timesteps": args.timesteps, "n_envs": n_envs,
        "wall_seconds": wall_seconds,
        "python": platform.python_version(),
        "torch": torch.__version__, "sb3": sb3.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_sha256": sha256(default_model_path()),
        "targets_sha256": sha256(default_targets_path()),
        "eval_seeds": EVAL_SEEDS[:args.eval_episodes],
        "smoke": bool(args.smoke),
    }
    if rnd_module is not None:
        manifest["rnd_intrinsic_tail"] = rnd_module.intrinsic_history[-5:]
    (out_dir / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"run": str(out_dir), "wall_seconds": wall_seconds, **eval_summary},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
