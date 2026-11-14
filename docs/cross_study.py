"""Cross-study comparison data (Reinforcement-Learning-Policy repo report).

Numbers supplied by the user (RLP experiment report, 11 pages, commit
d229af8): three evaluation columns (E1/E2/E3) on the RLP benchmark suite.
ACER is discrete-action only (SB3) so it cannot run on our continuous
16-D gesture tasks; it is included here as the discrete cross-reference.
"""

RLP_CROSS_STUDY = [
    # method,      E1,          E2,     E3,          note
    ("PPG",       "0.669±0.02", "0.681", "1.000±0.00", ""),
    ("PPO-KL",    "0.658",      "0.785", "0.992",      ""),
    ("PPO-RB",    "0.655",      "0.669", "0.995",      ""),
    ("PPO-Clip",  "0.652",      "0.620", "0.996",      "E2 种子散布 ±0.17"),
    ("ESPO",      "0.648",      "0.755", "0.568",      ""),
    ("TR-PPO-RB", "0.625",      "0.754", "0.694",      ""),
    ("TRPO",      "--",         "--",    "1.000",      "E3=1.000 真值；CG+线搜开销"),
    ("ACER",      "--",         "--",    "--",         "离散动作对照（SB3 ACER 仅支持离散，详见 RLP 报告）"),
]

SOURCE_NOTE = "来源：Reinforcement-Learning-Policy 仓实验报告（11 页，d229af8）"
