"""Aggregate the comparison matrix into summary.csv / summary.json.

Usage: python scripts/aggregate.py [--results results] [--out results]
Per run dir it reads: config.json, eval.json, episodes.csv (training curve),
ppo_diagnostics.json (when present).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ALGO_ORDER = ["ppo", "trpo", "ppo_kl", "td3", "sac", "ppo_rb", "ppo_espo",
              "ppo_ppg", "ppo_exo", "ppo_pbrs", "ppo_rnd"]
ALGO_LABEL = {
    "ppo": "PPO", "trpo": "TRPO", "ppo_kl": "PPO-KL", "td3": "TD3", "sac": "SAC",
    "ppo_rb": "TR-PPO-RB", "ppo_espo": "ESPO", "ppo_ppg": "PPG",
    "ppo_exo": "ExO-PPO", "ppo_pbrs": "PBRS+PPO", "ppo_rnd": "RND+PPO",
}


def reward_curve_metrics(episodes_path: Path):
    """Smoothed episode-reward AUC (normalized by final level) and steps to
    90% of the final smoothed reward. Training-time success is structurally
    ~0 under the sampled residual policy (hold criterion broken by noise),
    so reward curves are the informative training signal."""
    if not episodes_path.is_file():
        return float("nan"), None
    rows = list(csv.DictReader(episodes_path.open(encoding="utf-8")))
    if len(rows) < 10:
        return float("nan"), None
    ts = np.array([int(r["timesteps"]) for r in rows])
    rew = np.array([float(r["reward"]) for r in rows])
    k = min(50, len(rew))
    smooth = np.convolve(rew, np.ones(k) / k, mode="valid")
    ts_v = ts[k - 1:]
    lo, hi = float(smooth.min()), float(smooth.max())
    span = hi - lo
    if span < 1e-6:
        # Flat curve: no measurable learning progress; AUC degenerates to full
        # (curve is trivially at its final level); steps-to-90 undefined.
        return 1.0, None
    normalized = (smooth - lo) / span          # 0..1, bounded by construction
    auc = float(np.mean(normalized))
    target = lo + 0.9 * span
    hit = np.where(smooth >= target)[0]
    steps90 = int(ts_v[hit[0]]) if len(hit) else None
    return auc, steps90


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.results)
    out = Path(args.out or args.results)

    grouped: dict[tuple, list[dict]] = {}
    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_p, eval_p = run_dir / "config.json", run_dir / "eval.json"
        if not cfg_p.is_file() or not eval_p.is_file():
            continue
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        ev = json.loads(eval_p.read_text(encoding="utf-8"))
        auc, steps90 = reward_curve_metrics(run_dir / "episodes.csv")
        row = {
            "task": cfg["task"], "algo": cfg["algo"], "seed": cfg["seed"],
            "label": ALGO_LABEL.get(cfg["algo"], cfg["algo"]),
            "timesteps": cfg["timesteps"], "wall_seconds": cfg["wall_seconds"],
            "success_rate": ev["success_rate"],
            "finish_time": ev.get("mean_finish_time"),
            "finish_std": ev.get("std_finish_time"),
            "mean_reward": ev.get("mean_reward"),
            "penetration_mean_mm": ev.get("mean_max_penetration_mm"),
            "penetration_max_mm": ev.get("max_penetration_mm"),
            "train_auc": auc,
            "steps_to_90": steps90,
        }
        diag_p = run_dir / "ppo_diagnostics.json"
        if diag_p.is_file():
            diag = json.loads(diag_p.read_text(encoding="utf-8"))
            if diag:
                tail = diag[-max(10, len(diag) // 5):]
                row["clip_frac"] = float(np.mean([d["clip_frac"] for d in tail if d.get("clip_frac") is not None]))
                row["approx_kl"] = float(np.mean([d["approx_kl"] for d in tail if d.get("approx_kl") is not None]))
                row["ratio_max"] = float(np.max([d["ratio_max"] for d in diag if d.get("ratio_max") is not None]))
                stops = [d["stopped_epoch"] for d in diag if d.get("stopped_epoch") is not None]
                row["espo_stop_frac"] = len(stops) / len(diag)
                aux = [d["aux_kl"] for d in diag if d.get("aux_kl") is not None]
                if aux:
                    row["ppg_aux_kl"] = float(np.mean(aux))
        rows.append(row)
        grouped.setdefault((cfg["task"], cfg["algo"]), []).append(row)

    summary = []
    for (task, algo), runs in grouped.items():
        def agg(key):
            vals = [r[key] for r in runs if r.get(key) is not None and not (isinstance(r.get(key), float) and np.isnan(r.get(key)))]
            if not vals:
                return None, None
            return float(np.mean(vals)), float(np.std(vals))
        entry = {"task": task, "algo": algo, "label": ALGO_LABEL.get(algo, algo), "seeds": len(runs)}
        for key in ("success_rate", "finish_time", "mean_reward", "penetration_mean_mm",
                    "penetration_max_mm", "train_auc", "steps_to_90", "wall_seconds",
                    "clip_frac", "approx_kl", "ratio_max", "espo_stop_frac", "ppg_aux_kl"):
            mean, std = agg(key)
            entry[key] = mean
            entry[key + "_std"] = std
        summary.append(entry)

    def order(e):
        return (e["task"] != "free", ALGO_ORDER.index(e["algo"]))
    summary.sort(key=order)

    if rows:
        with (out / "runs.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    (out / "summary.json").write_text(
        json.dumps({"runs": len(rows), "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"aggregated {len(rows)} runs -> {out/'summary.csv'}")
    for e in summary:
        print(f'{e["task"]:>4} {e["label"]:<10} n={e["seeds"]} succ={e["success_rate"]:.3f}'
              f'±{(e["success_rate_std"] or 0):.3f} finish={e["finish_time"]}s auc={e["train_auc"]}')


if __name__ == "__main__":
    main()
