"""Generate the 12 comparison figures from results/ (publication style).

Usage: python scripts/make_figures.py --results results --out figures
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.5, "axes.grid": True,
    "grid.alpha": 0.25, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.6,
})
ALGO_ORDER = ["ppo", "trpo", "ppo_kl", "td3", "sac", "ppo_rb", "ppo_espo",
              "ppo_ppg", "ppo_exo", "ppo_pbrs", "ppo_rnd"]
LABEL = {"ppo": "PPO", "trpo": "TRPO", "ppo_kl": "PPO-KL", "td3": "TD3", "sac": "SAC", "ppo_rb": "TR-PPO-RB",
         "ppo_espo": "ESPO", "ppo_ppg": "PPG", "ppo_exo": "ExO-PPO",
         "ppo_pbrs": "PBRS+PPO", "ppo_rnd": "RND+PPO"}
COLOR = {"ppo": "#075f57", "trpo": "#7a4f9e", "ppo_kl": "#4a7d2f", "td3": "#8a5a44", "sac": "#5b6fa5", "ppo_rb": "#c2571f",
         "ppo_espo": "#a38bd0", "ppo_ppg": "#3d8a5f", "ppo_exo": "#c7a514",
         "ppo_pbrs": "#2f7fbf", "ppo_rnd": "#b04a6e"}
TASK_TITLE = {"free": "Task 1: free pose (collision-free)", "seq": "Task 2: V$\\to$OK$\\to$OPEN sequence"}


def smooth_curve(ts, succ, k=60):
    if len(succ) < k:
        k = max(5, len(succ) // 3)
    sm = np.convolve(succ, np.ones(k) / k, mode="valid")
    return ts[k - 1:], sm


def load_runs(root: Path):
    runs = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "eval.json").is_file():
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            ev = json.loads((d / "eval.json").read_text(encoding="utf-8"))
            episodes = list(csv.DictReader((d / "episodes.csv").open(encoding="utf-8")))
            diag = []
            dp = d / "ppo_diagnostics.json"
            if dp.is_file():
                diag = json.loads(dp.read_text(encoding="utf-8"))
            eval_rows = []
            er = d / "eval_episodes.csv"
            if er.is_file():
                eval_rows = list(csv.DictReader(er.open(encoding="utf-8")))
            runs.append({"dir": d, "cfg": cfg, "eval": ev, "episodes": episodes,
                         "diag": diag, "eval_rows": eval_rows})
    return runs


def grouped(runs, task):
    out = defaultdict(list)
    for r in runs:
        if r["cfg"]["task"] == task:
            out[r["cfg"]["algo"]].append(r)
    return out


def fig_learning(ax, runs, task, title):
    g = grouped(runs, task)
    for algo in ALGO_ORDER:
        curves = []
        for r in g.get(algo, []):
            eps = r["episodes"]
            if not eps:
                continue
            ts = np.array([int(e["timesteps"]) for e in eps])
            rew = np.array([float(e["reward"]) for e in eps])
            t, s = smooth_curve(ts, rew)
            curves.append((t, s))
        if not curves:
            continue
        n = min(len(c[0]) for c in curves)
        mean = np.mean([np.interp(np.linspace(c[0][0], c[0][-1], n), c[0], c[1]) for c in curves], axis=0)
        grid = np.linspace(0, 1, n)
        ax.plot(grid * curves[0][0][-1], mean, color=COLOR[algo], label=LABEL[algo])
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode reward (smoothed, 3 seeds)")
    ax.set_title(title, fontsize=10)


def bar_group(ax, summary, key, task, ylabel, fmt="{:.2f}", logy=False):
    algos = [a for a in ALGO_ORDER if any(e["algo"] == a and e["task"] == task for e in summary)]
    xs = np.arange(len(algos))
    means = [(_v if (_v := next((e[key] for e in summary if e["algo"] == a and e["task"] == task), None)) is not None else np.nan) for a in algos]
    stds = [next(((e.get(key + "_std") or 0.0) for e in summary if e["algo"] == a and e["task"] == task), 0) for a in algos]
    ax.bar(xs, means, yerr=stds, color=[COLOR[a] for a in algos], capsize=3, alpha=0.92)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[a] for a in algos], rotation=38, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    for x, m in zip(xs, means):
        if m == m:
            ax.text(x, m, fmt.format(m), ha="center", va="bottom", fontsize=6.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="hands-rl/results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    root, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    if not runs:
        raise SystemExit("no finished runs found in results/")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))["summary"] \
        if (root / "summary.json").is_file() else []
    if not summary:
        raise SystemExit("run scripts/aggregate.py first")

    def save(fig, name):
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png")
        plt.close(fig)
        print("fig:", name)

    # fig01/02 learning curves
    for name, task in (("fig01_learning_free", "free"), ("fig02_learning_seq", "seq")):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        fig_learning(ax, runs, task, TASK_TITLE[task])
        ax.legend(fontsize=7, ncol=3)
        save(fig, name)

    # fig03 final success bars (two panels)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, task in zip(axes, ("free", "seq")):
        bar_group(ax, summary, "success_rate", task, "success rate", "{:.2f}")
        ax.set_title(TASK_TITLE[task], fontsize=9.5)
        ax.set_ylim(0, 1.15)
    save(fig, "fig03_final_success")

    # fig04 finish time
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, task, ylab in zip(axes, ("free", "seq"),
                              ("arrival time (s)", "sequence time (s)")):
        bar_group(ax, summary, "finish_time", task, ylab, "{:.2f}")
        ax.set_title(TASK_TITLE[task], fontsize=9.5)
    save(fig, "fig04_finish_time")

    # fig05 sample efficiency
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for task, dx, hatch in (("free", -0.2, ""), ("seq", 0.2, "//")):
        algos = [a for a in ALGO_ORDER if any(e["algo"] == a and e["task"] == task for e in summary)]
        xs = np.arange(len(algos)) + dx
        vals = [(_v if (_v := next((e.get("steps_to_90") for e in summary if e["algo"] == a and e["task"] == task), None)) is not None else np.nan) for a in algos]
        ax.bar(xs, vals, width=0.38, color=[COLOR[a] for a in algos], alpha=0.9,
               hatch=hatch, label=TASK_TITLE[task].replace("$\\to$", ">"))
        for x, v in zip(xs, vals):
            if v == v:
                ax.text(x, v, f"{int(v/1000)}k", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(np.arange(len(ALGO_ORDER)))
    ax.set_xticklabels([LABEL[a] for a in ALGO_ORDER], rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("steps to 90% of final training reward")
    ax.legend(fontsize=8)
    save(fig, "fig05_sample_efficiency")

    # fig06 on-policy diagnostics
    keys = [("clip_frac", "clip fraction"), ("approx_kl", "approx KL"),
            ("ratio_max", "max ratio (training tail)"), ("train_auc", "training AUC")]
    ppo_variants = ["ppo", "ppo_rb", "ppo_espo", "ppo_ppg", "ppo_exo", "ppo_pbrs", "ppo_rnd"]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2))
    for ax, (key, title) in zip(axes.flat, keys):
        for task, dx in (("free", -0.19), ("seq", 0.19)):
            xs, vals, stds = [], [], []
            for i, a in enumerate(ppo_variants):
                e = next((e for e in summary if e["algo"] == a and e["task"] == task), None)
                if e and e.get(key) is not None:
                    xs.append(i + dx)
                    vals.append(e[key])
                    stds.append(e.get(key + "_std") or 0)
            ax.bar(xs, vals, yerr=stds, width=0.36, color=[COLOR[a] for a in ppo_variants[:len(xs):int(len(xs) > 1) or 1]], capsize=2)
        ax.set_xticks(range(len(ppo_variants)))
        ax.set_xticklabels([LABEL[a] for a in ppo_variants], rotation=40, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
    save(fig, "fig06_ppo_diagnostics")

    # fig07 ESPO vs PPO training KL trace + stop epochs
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    for algo in ("ppo", "ppo_espo"):
        g = grouped(runs, "seq").get(algo, [])
        for r in g[:1]:
            if r["diag"]:
                ts = [d["timesteps"] for d in r["diag"]]
                kl = [d["approx_kl"] for d in r["diag"]]
                ax.plot(ts, kl, color=COLOR[algo], label=LABEL[algo])
                stops = [(d["timesteps"], d["stopped_epoch"]) for d in r["diag"] if d.get("stopped_epoch") is not None]
                if stops:
                    ax.scatter([s[0] for s in stops], [0] * len(stops), marker="x",
                               color=COLOR[algo], s=22)
    ax.set_xlabel("steps"); ax.set_ylabel("approx KL / update")
    ax.set_title("Task 2 KL trace (x = ESPO early stop)", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1]
    g = grouped(runs, "seq").get("ppo_espo", [])
    stops = []
    for r in g:
        stops += [d["stopped_epoch"] for d in r["diag"] if d.get("stopped_epoch") is not None]
    if stops:
        ax.hist(stops, bins=range(1, 12), color=COLOR["ppo_espo"], alpha=0.85)
    ax.set_xlabel("stopped epoch index (of 10)")
    ax.set_ylabel("count")
    ax.set_title("ESPO early-stop distribution", fontsize=9)
    save(fig, "fig07_espo_stopping")

    # fig08 reward group curves + RND intrinsic trace
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    for algo in ("ppo", "ppo_pbrs", "ppo_rnd"):
        curves = []
        for r in grouped(runs, "seq").get(algo, []):
            eps = r["episodes"]
            if not eps:
                continue
            ts = np.array([int(e["timesteps"]) for e in eps])
            rew = np.array([float(e["reward"]) for e in eps])
            t, s = smooth_curve(ts, rew)
            curves.append((t, s))
        if curves:
            n = min(len(c[0]) for c in curves)
            mean = np.mean([np.interp(np.linspace(c[0][0], c[0][-1], n), c[0], c[1]) for c in curves], axis=0)
            ax.plot(np.linspace(0, 1, n) * curves[0][0][-1], mean, color=COLOR[algo], label=LABEL[algo])
    ax.set_xlabel("steps"); ax.set_ylabel("episode reward (smoothed)")
    ax.set_title("Reward-shaping group (Task 2)", fontsize=9); ax.legend(fontsize=8)
    ax = axes[1]
    rnd_runs = grouped(runs, "seq").get("ppo_rnd", [])
    for r in rnd_runs:
        cfg = r["cfg"]
        tail = cfg.get("rnd_intrinsic_tail") or []
        if tail:
            ax.plot(range(len(tail)), tail, alpha=0.7)
    ax.set_xlabel("last 5 rollouts (index)"); ax.set_ylabel("RND predictor loss")
    ax.set_title("RND intrinsic magnitude (tail, per seed)", fontsize=9)
    save(fig, "fig08_reward_group")

    # fig09 eval finish-time box plots
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, task in zip(axes, ("free", "seq")):
        data, labels = [], []
        for algo in ALGO_ORDER:
            vals = []
            for r in grouped(runs, task).get(algo, []):
                for e in r["eval_rows"]:
                    v = e["arrival_time"] or e["sequence_time"]
                    if v and e["is_success"] == "1":
                        vals.append(float(v))
            if vals:
                data.append(vals)
                labels.append(LABEL[algo])
        if data:
            bp = ax.boxplot(data, patch_artist=True, widths=0.55)
            for patch, algo in zip(bp["boxes"], [a for a in ALGO_ORDER][:len(data)]):
                patch.set_facecolor(COLOR[algo]); patch.set_alpha(0.75)
        ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
        ax.set_ylabel("time (s)")
        ax.set_title(TASK_TITLE[task], fontsize=9.5)
    save(fig, "fig09_eval_box")

    # fig10 phase decomposition (task 2)
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    algos = [a for a in ALGO_ORDER if any(e["algo"] == a and e["task"] == "seq" for e in summary)]
    xs = np.arange(len(algos))
    parts = np.zeros(len(algos))
    phase_names = ["V phase", "OK phase", "OPEN phase"]
    phase_colors = ["#c2571f", "#2f7fbf", "#075f57"]
    for i in range(3):
        vals = []
        for a in algos:
            e = next((e for e in summary if e["algo"] == a and e["task"] == "seq"), None)
            v = np.nan
            if e is not None:
                per_seed = []
                for r in grouped(runs, "seq").get(a, []):
                    if r["eval"].get("mean_phase_times"):
                        per_seed.append(r["eval"]["mean_phase_times"][i])
                if per_seed:
                    v = float(np.mean(per_seed))
            vals.append(0.0 if v != v else v)
        ax.bar(xs, vals, bottom=parts, color=phase_colors[i], label=phase_names[i], alpha=0.9)
        parts += np.array(vals)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[a] for a in algos], rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("mean phase time (s, successful eps)")
    ax.legend(fontsize=8)
    ax.set_title("Task 2 phase decomposition", fontsize=10)
    save(fig, "fig10_phase_decomposition")

    # fig11 wall-clock efficiency scatter
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for task, marker in (("free", "o"), ("seq", "s")):
        for e in summary:
            if e["task"] == task and e.get("success_rate") is not None:
                ax.errorbar(e.get("wall_seconds", 0) / 60, e["success_rate"],
                            yerr=e.get("success_rate_std") or 0, fmt=marker,
                            color=COLOR[e["algo"]], ms=5, capsize=2)
    for algo in ALGO_ORDER:
        ax.scatter([], [], color=COLOR[algo], label=LABEL[algo], s=18)
    ax.set_xlabel("wall-clock training time (min, 150k steps)")
    ax.set_ylabel("final success rate")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("o = Task 1, s = Task 2", fontsize=9)
    save(fig, "fig11_wallclock")

    # fig12 safety: penetration bars
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, task in zip(axes, ("free", "seq")):
        bar_group(ax, summary, "penetration_mean_mm", task, "eval mean max-penetration (mm)", "{:.2f}")
        ax.set_title(TASK_TITLE[task], fontsize=9.5)
    save(fig, "fig12_safety_penetration")

    # fig13 cross-study (RLP benchmark, static data from user's report)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from cross_study import RLP_CROSS_STUDY
    import matplotlib.pyplot as _plt2
    fig, ax = _plt2.subplots(figsize=(7.4, 3.4))
    methods = [m for m, *_ in RLP_CROSS_STUDY if m != "ACER"]
    x = np.arange(len(methods))
    width = 0.26
    palette = ["#2f7fbf", "#075f57", "#c2571f"]
    for j, col in enumerate(["E1", "E2", "E3"]):
        vals = []
        for row in RLP_CROSS_STUDY:
            if row[0] == "ACER":
                continue
            v = row[1 + j].split("±")[0]
            vals.append(float(v) if v != "--" else np.nan)
        ax.bar(x + (j - 1) * width, vals, width, label=col, color=palette[j], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RLP benchmark score")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8)
    ax.set_title("Cross-study: RLP benchmark (source: RLP report d229af8; ACER discrete-only)", fontsize=9)
    save(fig, "fig13_cross_study")

    # fig14 negative-result attribution (TRPO drift + RND lambda)
    import json as _json
    attr_dir = root / "../results_attribution/attribution_summary.json"
    if attr_dir.is_file():
        attr = _json.loads(attr_dir.read_text(encoding="utf-8"))
        fig, axes = _plt2.subplots(1, 2, figsize=(9.6, 3.6))
        ax = axes[0]
        styles = {"free_trpo_s1_trpo500": ("#7a4f9e", "-"), "free_trpo_s2_trpo500": ("#9a6fbe", "--"),
                  "seq_trpo_s1_trpo500": ("#b99ad6", ":")}
        for name, (c, ls) in styles.items():
            cur = attr["trpo_default_drift"].get(name)
            if cur:
                xs = sorted(int(k) for k in cur)
                ax.plot(xs, [cur[str(x)] for x in xs], color=c, ls=ls, marker="o", ms=3,
                        label=name.replace("_trpo", " TRPO").replace("_s", " s").replace("_trpo500", " 500k"))
        low = attr["trpo_lowstd_drift"].get("free_trpo_s1_trpolowstd")
        if low:
            xs = sorted(int(k) for k in low)
            ax.plot(xs, [low[str(x)] for x in xs], color="#c2571f", marker="s", ms=4,
                    label="free s1 matched-init (150k)")
        lowq = attr["trpo_lowstd_drift"].get("seq_trpo_s1_trpolowstd")
        if lowq:
            xs = sorted(int(k) for k in lowq)
            ax.plot(xs, [lowq[str(x)] for x in xs], color="#a05018", ls="--", marker="s", ms=4,
                    label="seq s1 matched-init (150k)")
        ax.set_xlabel("checkpoint steps"); ax.set_ylabel("success rate (25-seed det. eval)")
        ax.set_ylim(-0.03, 1.05); ax.legend(fontsize=6.2, loc="upper right")
        ax.set_title("TRPO drift curves: default init flat-zero; matched init rescued then eroded", fontsize=8.5)
        ax = axes[1]
        labels, vals, cols = [], [], []
        for task in ("free", "seq"):
            labels += [task + " λ=0.5", task + " λ=0.05"]
            m50 = 0.0
            vals += [m50, attr["rnd_lambda"].get(f"{task}_ppo_rnd_s1_rnd005", 0.0)]
            cols += ["#b04a6e", "#d98aa5"]
        ax.bar(range(4), vals, color=cols)
        ax.axhline(1.0, color="#075f57", ls=":", lw=1)
        ax.text(3.4, 1.02, "PPO baseline", color="#075f57", fontsize=7)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("success rate"); ax.set_ylim(0, 1.15)
        ax.set_title("RND lambda ablation: dual-lambda zeros = direction conflict", fontsize=9)
        save(fig, "fig14_attribution")

    print("ALL_FIGURES_DONE")


if __name__ == "__main__":
    main()
