"""Integrate cross-study + TRPO/PPO-KL/ACER into site, paper, textbook."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CROSS_TABLE_MD = """| 方法 | E1 | E2 | E3 | 注记 |
|---|---|---|---|---|
| PPG | 0.669±0.02 | 0.681 | 1.000±0.00 | |
| PPO-KL | 0.658 | 0.785 | 0.992 | |
| PPO-RB | 0.655 | 0.669 | 0.995 | |
| PPO-Clip | 0.652 | 0.620 | 0.996 | E2 种子散布 ±0.17 |
| ESPO | 0.648 | 0.755 | 0.568 | |
| TR-PPO-RB | 0.625 | 0.754 | 0.694 | |
| TRPO | -- | -- | 1.000 | E3=1.000 真值；CG+线搜开销 |
| ACER | -- | -- | -- | 离散对照（SB3 ACER 仅离散动作） |"""

EDITS = {
    "scripts/gen_site.py": [
        # experiments page: cross-study card before reproduce section
        ('<h2>5 · 复现命令</h2>',
         '<h2>5 · 跨研究对照（RLP 基准）</h2>\n<div class="card">\n'
         + CROSS_TABLE_MD.replace("\n", "\n") +
         '\n<p class="figcap">来源：Reinforcement-Learning-Policy 仓实验报告（11 页，d229af8）。'
         'ACER 在 SB3 中仅支持离散动作空间，无法在本工程连续 16 维手势任务上实跑，作为离散对照与分类引用保留。</p>\n'
         '<img class="fig" src="figs/fig13_cross_study.png" alt="cross-study">'
         '<p class="figcap">fig13 · 跨研究对比图</p></div>\n<h2>6 · 复现命令</h2>'),
        # algos16 page: role table row updates + new rows
        ('<tr><td><b>B PPO 算法改进</b></td><td>06 TR-PPO-RB · 07 ESPO · 08 PPG · 25 ExO-PPO · 26 OPPO</td>',
         '<tr><td><b>B PPO 算法改进</b></td><td>01 TRPO(CG+线搜) · PPO-KL(自适应惩罚) · 06 TR-PPO-RB · 07 ESPO · 08 PPG · 25 ExO-PPO · 26 OPPO</td>'),
        ('<td><b>06/07/08/25 实跑</b>；26 引用（无 LLM 服务环）</td>',
         '<td><b>TRPO/PPO-KL/06/07/08/25 实跑</b>；26 引用（无 LLM 服务环）</td>'),
        ('<tr><td><b>D 落地参照</b></td><td>31 TD3 · 32 SAC · 56 ANYmal · 55 RMA · 62 RL-100</td>',
         '<tr><td><b>D 落地参照</b></td><td>57 ACER(离散对照) · 31 TD3 · 32 SAC · 56 ANYmal · 55 RMA · 62 RL-100</td>'),
        ('<td>31/32 实跑基线；其余参照</td>',
         '<td>31/32 实跑基线；ACER 因 SB3 仅离散无法连续实跑（见跨研究表）；其余参照</td>'),
        # KPI hero: 9->11 already patched; also algos16 hero subtitle
        ('四组演进阶梯 · 实跑/引用角色划分 · 目标函数几何对比',
         '四组演进阶梯 · 11 算法实跑（含 TRPO/PPO-KL）· ACER 离散对照 · 目标函数几何'),
    ],
    "paper/ieee_main.tex": [
        ("evaluate three algorithm families---PPO, TD3, and SAC---and six published\nPPO-centric improvements (TR-PPO-RB rollback, ESPO early stopping, PPG phasic\nupdates, ExO-PPO smooth off-policy surrogates, potential-based reward shaping,\nand RND intrinsic motivation) under a fixed budget",
         "evaluate eleven methods---TRPO with conjugate-gradient line search, PPO and\nits adaptive-KL variant, TD3, SAC, and six published PPO-centric improvements\n(TR-PPO-RB rollback, ESPO early stopping, PPG phasic updates, ExO-PPO smooth\noff-policy surrogates, potential-based reward shaping, and RND intrinsic\nmotivation) under a fixed budget"),
        ("\\item a controlled study of nine algorithms---three families and six\npublished PPO improvements---under identical rollouts, budgets, networks,\nand evaluation seeds (Sec.~\\ref{sec:results}), with all reimplementation\ndeviations stated explicitly.",
         "\\item a controlled study of eleven algorithms---TRPO, PPO, PPO-KL, TD3,\nSAC, and six published PPO improvements---under identical rollouts, budgets,\nnetworks, and evaluation seeds (Sec.~\\ref{sec:results}), with all\nreimplementation deviations stated explicitly."),
        ("01--05 & TRPO, GAE, PPO, Impl., OnPolicy-AC & foundations; 03 run as baseline; 04--05 define protocol \\\\",
         "01--05 & TRPO, GAE, PPO, Impl., OnPolicy-AC & foundations; 01 (CG+line search) and 03 run; PPO-KL variant run; 04--05 define protocol \\\\"),
        ("56, 55, 62 & ANYmal, RMA, RL-100 & deployment references \\\\",
         "57 & ACER & discrete-action cross-reference (SB3 ACER cannot run continuous tasks) \\\\\n56, 55, 62 & ANYmal, RMA, RL-100 & deployment references \\\\"),
        ("PPO uses standard published hyperparameters; TD3 and SAC update in batches",
         "TRPO (sb3-contrib) uses a full-batch conjugate-gradient update with target\nKL 0.01, CG damping 0.1, and backtracking line search; PPO-KL implements the\nadaptive-KL-penalty objective of the original PPO paper ($\\beta$ doubled or\nhalved when KL leaves $[0.5,1.5]\\times$ target 0.008). PPO uses standard\npublished hyperparameters; TD3 and SAC update in batches"),
        ("\\section{Results and Discussion}\\label{sec:results}\n\\input{results_auto}",
         "\\section{Results and Discussion}\\label{sec:results}\n\\input{results_auto}\n\n\\subsection{Cross-Study Context}\nAs external context, Table~\\ref{tab:cross} quotes the companion\nReinforcement-Learning-Policy study (report d229af8) that benchmarks several\nof the same PPO variants on its own suite; TRPO attains the exact E3 optimum\nat the cost of conjugate-gradient line-search overhead, and ACER serves as a\ndiscrete-action cross-reference that cannot enter our continuous tasks.\n\n\\begin{table}[!t]\n\\caption{Cross-study scores (RLP benchmark, E1/E2/E3)}\n\\label{tab:cross}\n\\centering\\small\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\nMethod & E1 & E2 & E3 \\\\\n\\midrule\nPPG & 0.669$\\pm$0.02 & 0.681 & 1.000$\\pm$0.00 \\\\\nPPO-KL & 0.658 & 0.785 & 0.992 \\\\\nPPO-RB & 0.655 & 0.669 & 0.995 \\\\\nPPO-Clip & 0.652 & 0.620 & 0.996$^{\\pm0.17}$ \\\\\nESPO & 0.648 & 0.755 & 0.568 \\\\\nTR-PPO-RB & 0.625 & 0.754 & 0.694 \\\\\nTRPO & -- & -- & 1.000 \\\\\nACER & \\multicolumn{3}{c}{discrete-only cross-reference} \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}"),
    ],
    "textbook/ch6_rl.tex": [
        ("01 TRPO & 基础与复现 & 旧策略轨迹上估计优势；样本平均 KL 约束 + 共轭梯度 + 回溯线搜索 & 理论基线（对照章） \\\\",
         "01 TRPO & 基础与复现 & 旧策略轨迹上估计优势；样本平均 KL 约束 + 共轭梯度 + 回溯线搜索 & \\textbf{基线 trpo（实跑，CG+线搜）} \\\\"),
        ("03 PPO & 基础与复现 & $\\min(\\rho\\hat A,\\mathrm{clip}(\\rho)\\hat A)$；$\\varepsilon{=}0.2$ 得 0.82 归一分 & \\textbf{基线（实跑）} \\\\",
         "03 PPO & 基础与复现 & $\\min(\\rho\\hat A,\\mathrm{clip}(\\rho)\\hat A)$；$\\varepsilon{=}0.2$ 得 0.82 归一分 & \\textbf{基线（实跑）} \\\\\n"
         "— PPO-KL & PPO 变体 & 自适应 KL 惩罚 $\\rho\\hat A-\\beta\\,\\mathrm{KL}$；$\\beta$ 随 KL 出 $[0.5,1.5]\\times$目标 倍增/减半 & \\textbf{变体 ppo\\_kl（实跑）} \\\\\n"
         "57 ACER & 连续控制 & 截断重要性比率 + Retrace + 回放纠偏；SB3 实现仅离散动作 & 分类引用（离散对照，见跨研究表） \\\\"),
        ("\\subsection{三种基线算法一页纸}",
         "\\subsection{跨研究对照（RLP 基准）}\n同源方法在姊妹仓 Reinforcement-Learning-Policy 的自有基准上另有三列评测（E1/E2/E3）：\nPPG 0.669$\\pm$0.02/0.681/1.000$\\pm$0.00，PPO-KL 0.658/0.785/0.992，PPO-RB 0.655/0.669/0.995，\nPPO-Clip 0.652/0.620/0.996（E2 种子散布 $\\pm$0.17），ESPO 0.648/0.755/0.568，TR-PPO-RB 0.625/0.754/0.694，\nTRPO E3=1.000 真值（代价为 CG+线搜开销）；ACER 为离散对照。两基准任务不同，数值不可直接横比，仅作方法族参照。\n\n\\subsection{三种基线算法一页纸}"),
    ],
}


def main():
    for rel, pairs in EDITS.items():
        path = ROOT / rel
        s = path.read_text(encoding="utf-8")
        for old, new in pairs:
            if old in s:
                s = s.replace(old, new, 1)
            else:
                print(f"MISS ({rel}): {old[:56]!r}")
        path.write_text(s, encoding="utf-8", newline="\n")
        print("patched", rel)


if __name__ == "__main__":
    main()
