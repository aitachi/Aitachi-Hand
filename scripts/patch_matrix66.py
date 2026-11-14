"""Bump protocol wording from 9 methods / 54 runs to 11 methods / 66 runs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPL = [
    ("scripts/gen_readme.py", [
        ("2 tasks × 9 methods × 3 seeds = 54 runs", "2 tasks × 11 methods × 3 seeds = 66 runs"),
        ("2 任务 × 9 方法 × 3 种子 = 54 跑", "2 任务 × 11 方法 × 3 种子 = 66 跑"),
        ("2タスク × 9手法 × 3シード = 54実行", "2タスク × 11手法 × 3シード = 66実行"),
        ("2과제 × 9방법 × 3시드 = 54실행", "2과제 × 11방법 × 3시드 = 66실행"),
        ("full 54-run matrix (~6-8 h, RTX 3070)", "full 66-run matrix (~6-8 h, RTX 3070)"),
        ("54 跑全矩阵（约 6-8 小时，RTX 3070）", "66 跑全矩阵（约 6-8 小时，RTX 3070）"),
        ("54実行の全行列(~6-8時間)", "66実行の全行列(~6-8時間)"),
        ("54실행 전체 매트릭스(~6-8시간)", "66실행 전체 매트릭스(~6-8시간)"),
        ("bash scripts/run_matrix_all.sh          # 66 跑全矩阵", "bash scripts/run_matrix_all.sh"),
    ]),
    ("scripts/gen_site.py", [
        ('<div><b>9</b><span>算法（3 基线 + 6 改进变体）</span></div>',
         '<div><b>11</b><span>算法（PPO 系 8 + TD3/SAC）</span></div>'),
        ('<div><b>54</b><span>实验跑（2×9×3 · 已完成 {n}）</span></div>',
         '<div><b>66</b><span>实验跑（2×11×3 · 已完成 {n}）</span></div>'),
        ("比较 PPO / TD3 / SAC 与 TR-PPO-RB · ESPO · PPG · ExO-PPO · PBRS · RND，",
         "比较 PPO / TRPO / PPO-KL / TD3 / SAC 与 TR-PPO-RB · ESPO · PPG · ExO-PPO · PBRS · RND，"),
        ("（任务一 54 跑矩阵：2 任务 × 9 算法 × 3 种子）", ""),
        ("bash scripts/run_matrix_all.sh           # 54 跑全矩阵 ~6-8h（RTX 3070）",
         "bash scripts/run_matrix_all.sh           # 66 跑全矩阵 ~6-8h（RTX 3070；trpo/ppo_kl 已并入）"),
        ("矩阵：2 任务 × 9 方法 × 3 种子 = 54 跑 × 150k 步；",
         "矩阵：2 任务 × 11 方法 × 3 种子 = 66 跑 × 150k 步（含 TRPO CG+线搜、PPO-KL 自适应惩罚）；"),
        ("- **矩阵**：2 任务 × 9 算法", "- **矩阵**：2 任务 × 11 算法"),
        ("9算法×2任务×种子1-3 = 54 全集", "11算法×2任务×种子1-3 = 66 全集"),
        ("<b>54</b>", "<b>66</b>"),
    ]),
    ("scripts/push_github.py", [
        ('INCLUDE_FILES = ["README.md", ".gitignore", "LICENSE", "requirements.txt"]',
         'INCLUDE_FILES = ["README.md", "README.zh.md", "README.ja.md", "README.ko.md",\n                  ".gitignore", "LICENSE", "requirements.txt"]'),
    ]),
    ("textbook/ch6_rl.tex", [
        ("\\textbf{矩阵}：2 任务 $\\times$ 9 算法", "\\textbf{矩阵}：2 任务 $\\times$ 11 算法（含 TRPO 与 PPO-KL）"),
        ("ppo, td3, sac, ppo\\_rb, ppo\\_espo, ppo\\_ppo(ppg), ppo\\_exo, ppo\\_pbrs, ppo\\_rnd",
         "ppo, trpo, ppo\\_kl, td3, sac, ppo\\_rb, ppo\\_espo, ppo\\_ppg, ppo\\_exo, ppo\\_pbrs, ppo\\_rnd"),
    ]),
]


def main():
    for rel, pairs in REPL:
        path = ROOT / rel
        s = path.read_text(encoding="utf-8")
        for old, new in pairs:
            if old in s:
                s = s.replace(old, new)
            else:
                print(f"miss ({rel}): {old[:48]!r}")
        path.write_text(s, encoding="utf-8", newline="\n")
        print("patched", rel)


if __name__ == "__main__":
    main()
