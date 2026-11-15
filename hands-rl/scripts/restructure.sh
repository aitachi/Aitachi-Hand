#!/usr/bin/env bash
# Phase-B atomic repository restructure (run ONLY after the matrix finishes:
# no run_matrix/matrix_worker processes may be alive).
# Layout after this script:
#   hands-rl/         envs/ algorithms/ scripts/(training+analysis) results/ logs/
#   hands-sim/        fetch_assets.py + mujoco/isaac command guides
#   hands-hardware/   calibration tables + servo protocol + poses
#   textbook/ paper/ figures/ README* LICENSE  (root deliverables, unchanged)
set -e
cd "$(dirname "$0")/../.."

if ps -e --no-headers -o cmd= | grep -qE "run_matrix[.]py|matrix_worker[.]sh"; then
  echo "REFUSED: training processes still alive - run after matrix completion"; exit 1
fi

mkdir -p hands-rl hands-sim hands-hardware

# --- hands-rl ---
[ -d envs ] && git mv envs hands-rl/envs 2>/dev/null || mv envs hands-rl/envs
[ -d algorithms ] && mv algorithms hands-rl/algorithms
mkdir -p hands-rl/scripts
for f in run_matrix.py aggregate.py make_figures.py gen_readme.py \
         gen_results_snippet.py gen_site.py matrix_worker.sh matrix_worker_all.sh \
         smoke_all.sh smoke_trpo_kl.sh regen_all.sh finalize.sh restructure.sh \
         health_check.sh cpu_delta.sh kill_run.sh stack_dump.sh install_contrib.sh \
         push_github.py fetch_assets.py; do
  [ -f "scripts/$f" ] && mv "scripts/$f" "hands-rl/scripts/$f"
done
# patch scripts still referenced at old path by docs -> drop empty old dirs
[ -d scripts ] && rmdir scripts 2>/dev/null || true
[ -d results ] && mv results hands-rl/results
[ -d logs ] && mv logs hands-rl/logs
[ -f docs/cross_study.py ] && mv docs/cross_study.py hands-rl/cross_study.py

# --- hands-sim ---
cat > hands-sim/README.md <<'EOF'
# hands-sim — 仿真教程与资产

## MuJoCo
```bash
pip install mujoco==3.3.7
python -m mujoco.viewer --mjcf=<orca_v2>/scene_right.xml   # 官方描述仓 v2 场景
# RL 环境（hands-rl/envs）：AitachiFree-v1 / AitachiSeq-v1
```
执行器 kp=12/kv=0.5（腕 60/3）· 物理 0.002s · 控制 0.02s · 隐式快速积分。

## Isaac Sim / Isaac Lab（Ubuntu·WSL 无界面管线）
```bash
export AHAND_ROOT=/mnt/c/Users/ASUS/Desktop/AHand/gpu-validation
export AHAND_PY="$HOME/isaac_sim/venv/bin/python"
bash ./run_isaac_orca_wsl.sh        # 17 关节 GPU 动力学（600 步 · 误差 0.0178 rad）
bash ./run_isaac_cylinder_handoff_wsl.sh && tail -n 35 logs/handoff-latest.log
```
详见教材第 5.2 章。资产下载：hands-rl/scripts/fetch_assets.py。

## 模型获取
`python hands-rl/scripts/fetch_assets.py`（111MB 官方描述仓，不入 git）或设 `AHAND_MODEL_ROOT`。
EOF

# --- hands-hardware ---
cat > hands-hardware/README.md <<'EOF'
# hands-hardware — 硬件与标定（实测口径）

- `calibration_axes.txt`：17 轴 min/center/max 实测标定表（FD.exe 缓动采集）
- 舵机协议（帧格式/寄存器 41·40·56/0x83 同步写/四重护栏）：教材附录 A
- 轴映射：拇13/14/15/16(14=主屈) · 食6/7/8 · 中5/12/4 · 无名1/2/3 · 小9/10/11 · 腕0
- 安装/验收/避坑：教材第 1–2 章
EOF
[ -f docs/calibration_axes.txt ] && mv docs/calibration_axes.txt hands-hardware/
[ -d docs ] && rmdir docs 2>/dev/null || true

# --- path fixes inside moved scripts (they resolve ROOT as parents[1]=hands-rl) ---
# gen_readme/gen_site/gen_results_snippet/make_figures write to ROOT-relative paths:
# point them at the repo root (parents[2]).
python - <<'PY'
from pathlib import Path
fixes = {
    "hands-rl/scripts/gen_readme.py": [("ROOT = Path(__file__).resolve().parents[1]",
                                        "ROOT = Path(__file__).resolve().parents[2]")],
    "hands-rl/scripts/gen_site.py": [("ROOT = Path(__file__).resolve().parents[1]",
                                      "ROOT = Path(__file__).resolve().parents[2]")],
    "hands-rl/scripts/make_figures.py": [],
    "hands-rl/scripts/run_matrix.py": [
        ("sys.path.insert(0, str(Path(__file__).resolve().parents[1]))",
         "sys.path.insert(0, str(Path(__file__).resolve().parents[2]))")],
    "hands-rl/scripts/push_github.py": [
        ("ROOT = Path(__file__).resolve().parents[1]",
         "ROOT = Path(__file__).resolve().parents[2]"),
        ('INCLUDE_DIRS = ["envs", "algorithms", "scripts", "textbook", "paper", "figures", "docs", "results"]',
         'INCLUDE_DIRS = ["hands-rl/envs", "hands-rl/algorithms", "hands-rl/scripts",\n                 "hands-rl/results", "hands-rl/cross_study.py",\n                 "hands-sim", "hands-hardware", "textbook", "paper", "figures"]')],
    "hands-rl/scripts/regen_all.sh": [("$PY scripts/", "$PY hands-rl/scripts/")],
    "hands-rl/scripts/finalize.sh": [("$PY scripts/", "$PY hands-rl/scripts/")],
    "hands-rl/cross_study.py": [],
}
for rel, pairs in fixes.items():
    p = Path(rel)
    if not p.is_file():
        continue
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")
    print("paths fixed:", rel)
PY

echo "RESTRUCTURE_DONE"
ls -d hands-rl hands-sim hands-hardware textbook paper figures
