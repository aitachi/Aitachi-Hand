"""Render paper/results_auto.tex (results tables + key numbers) from results/summary.json.

Regenerate after the matrix completes so the IEEE paper always cites exact data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL = {"ppo": "PPO", "td3": "TD3", "sac": "SAC", "ppo_rb": "TR-PPO-RB",
         "ppo_espo": "ESPO", "ppo_ppg": "PPG", "ppo_exo": "ExO-PPO",
         "ppo_pbrs": "PBRS+PPO", "ppo_rnd": "RND+PPO"}
ORDER = list(LABEL)


def fmt(v, digits=3, dash="--"):
    if v is None:
        return dash
    try:
        if v != v:
            return dash
    except TypeError:
        return dash
    return f"{v:.{digits}f}" if digits else f"{v}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="paper/results_auto.tex")
    args = parser.parse_args()
    summary = json.loads((Path(args.results) / "summary.json").read_text(encoding="utf-8"))["summary"]
    by = {(e["task"], e["algo"]): e for e in summary}
    n_runs = json.loads((Path(args.results) / "summary.json").read_text(encoding="utf-8"))["runs"]

    lines = []
    lines.append(f"% AUTO-GENERATED from summary.json ({n_runs} runs). Do not edit by hand.")
    lines.append("\\subsection{Quantitative Results}")
    for task, name in (("free", "Task 1: collision-free free posing"),
                       ("seq", "Task 2: V$\\to$OK$\\to$OPEN sequencing")):
        lines.append("\\begin{table}[!t]")
        lines.append(f"\\caption{{{name}: mean$\\pm$std over seeds (25-episode deterministic eval)}}")
        lines.append(f"\\label{{tab:res-{task}}}")
        lines.append("\\centering\\small")
        lines.append("\\begin{tabular}{@{}lccc@{}}")
        lines.append("\\toprule")
        lines.append("Method & Success & Time (s) & Pen. (mm) \\\\")
        lines.append("\\midrule")
        for algo in ORDER:
            e = by.get((task, algo))
            if not e:
                continue
            lines.append(
                f"{LABEL[algo]} & {fmt(e['success_rate'])}$\\pm${fmt(e.get('success_rate_std'), 3)} & "
                f"{fmt(e.get('finish_time'), 2)} & {fmt(e.get('penetration_mean_mm'), 2)} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

    best_free = max((e for e in summary if e["task"] == "free"), key=lambda e: e["success_rate"], default=None)
    best_auc = max((e for e in summary if e["task"] == "seq" and e.get("train_auc") is not None),
                   key=lambda e: e["train_auc"], default=None)
    lines.append("\\subsection{Key Findings}")
    if best_free:
        lines.append(
            f"On Task 1, {LABEL[best_free['algo']]} attains the highest deterministic success rate "
            f"({fmt(best_free['success_rate'])}) with mean arrival {fmt(best_free.get('finish_time'), 2)}\\,s; "
            "Fig.~\\ref{fig:learn-free}--\\ref{fig:safety} report learning curves, sample efficiency, "
            "phase decomposition, wall-clock efficiency, and safety penetration.")
    if best_auc:
        lines.append(
            f"On Task 2, {LABEL[best_auc['algo']]} shows the strongest training AUC ({fmt(best_auc['train_auc'])}); "
            "ESPO early-stop statistics and PPG auxiliary KL appear in the diagnostics figure; "
            "all numbers regenerate from \\code{results/summary.json} via \\code{scripts/gen\\_results\\_snippet.py}.")

    lines.append("\\begin{figure}[!t]\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{fig01_learning_free.pdf}")
    lines.append("\\caption{Task 1 training curves (smoothed success rate vs.\\ environment steps; 3 seeds).}\\label{fig:learn-free}")
    lines.append("\\end{figure}")
    lines.append("")
    lines.append("\\begin{figure}[!t]\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{fig02_learning_seq.pdf}")
    lines.append("\\caption{Task 2 training curves.}\\label{fig:learn-seq}")
    lines.append("\\end{figure}")
    lines.append("")
    lines.append("\\begin{figure}[!t]\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{fig05_sample_efficiency.pdf}")
    lines.append("\\caption{Sample efficiency: steps for the smoothed training reward to reach 90\% of its final level.}\\label{fig:eff}")
    lines.append("\\end{figure}")
    lines.append("")
    lines.append("\\begin{figure}[!t]\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{fig10_phase_decomposition.pdf}")
    lines.append("\\caption{Task 2 per-phase time decomposition on successful episodes.}\\label{fig:phase}")
    lines.append("\\end{figure}")
    lines.append("")
    lines.append("\\begin{figure}[!t]\\centering")
    lines.append("\\includegraphics[width=\\linewidth]{fig12_safety_penetration.pdf}")
    lines.append("\\caption{Safety proxy: mean maximum self-penetration during evaluation.}\\label{fig:safety}")
    lines.append("\\end{figure}")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({n_runs} runs)")


if __name__ == "__main__":
    main()
