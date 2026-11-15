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
