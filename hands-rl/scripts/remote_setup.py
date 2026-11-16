"""Remote: install deps + smoke test on seetacloud 3090."""
import paramiko

HOST, PORT, USER, PWD = "connect.nmb2.seetacloud.com", 17281, "root", "zC2ydRyLCjK9"
PY = "/root/miniconda3/bin/python"

SETUP = r"""
source /etc/network_turbo
{PY} -m pip install -q gymnasium==1.4.0 mujoco==3.3.7 stable_baselines3==2.9.2 \
  sb3-contrib==2.9.0 matplotlib pandas 2>&1 | tail -2
cd /root/autodl-tmp/ahand
mkdir -p /root/autodl-tmp/output
cp model/gesture_targets.json /root/autodl-tmp/output/
export AHAND_MODEL_ROOT=/root/autodl-tmp/ahand/model/orca_v2
PYTHONPATH=hands-rl {PY} - <<'PYEOF'
import gymnasium as gym
import envs
e = gym.make("AitachiSeq-v1")
obs, info = e.reset(seed=1)
print("ENV_OK", e.action_space.shape, e.observation_space.shape)
PYEOF
PYTHONPATH=hands-rl {PY} hands-rl/scripts/run_matrix.py --task free --algo ppo \
  --seed 9 --timesteps 2048 --out /tmp/rsmoke --eval-episodes 3 --smoke 2>/dev/null | tail -1
""".replace("{PY}", PY)


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PWD, timeout=20)
    _, out, err = cli.exec_command(SETUP, timeout=1500)
    print((out.read() + err.read()).decode(errors="replace")[-3000:])
    cli.close()


if __name__ == "__main__":
    main()
