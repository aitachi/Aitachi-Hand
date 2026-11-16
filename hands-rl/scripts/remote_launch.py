"""Launch the completion batch on the seetacloud 3090 (two parallel workers)."""
import paramiko

HOST, PORT, USER, PWD = "connect.nmb2.seetacloud.com", 17281, "root", "zC2ydRyLCjK9"
PY = "/root/miniconda3/bin/python"
BASE = "/root/autodl-tmp/ahand"

WORKER = r"""#!/usr/bin/env bash
# server-side worker: sequential list, claim+skip semantics
set -u
cd /root/autodl-tmp/ahand
export AHAND_MODEL_ROOT=/root/autodl-tmp/ahand/model/orca_v2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/root/miniconda3/bin/python
run() { # outdir task algo seed steps tag extra...
  local outdir=$1 task=$2 algo=$3 seed=$4 steps=$5 tag=$6; shift 6
  local dir="$outdir/${task}_${algo}_s${seed}_${tag}"
  [ -f "$dir/eval.json" ] && { echo "[skip] $dir"; return; }
  mkdir -p "$dir"; echo $$ > "$dir/claim"
  echo "[run ] $(date +%H:%M:%S) $dir"
  PYTHONPATH=hands-rl $PY hands-rl/scripts/run_matrix.py --task "$task" --algo "$algo" \
    --seed "$seed" --timesteps "$steps" --out "$outdir" --eval-episodes 25 \
    --tag "$tag" "$@" > "$outdir/../attrlog_${task}_${algo}_s${seed}_${tag}.log" 2>&1
  echo "[done] $(date +%H:%M:%S) exit=$? $dir"
  rm -f "$dir/claim"
}
ATTR=hands-rl/results_attribution
RES=hands-rl/results
mkdir -p "$ATTR" "$RES"
"$@"
echo "REMOTE_WORKER_DONE"
"""

LAUNCH = r"""
cd /root/autodl-tmp/ahand
cat > /tmp/w_a.sh <<'EOS'
__WORKER__
EOS
# worker A: attribution completion (7 runs)
nohup bash /tmp/w_a.sh \
  run hands-rl/results_attribution free ppo_rnd 1 150000 rnd000 --rnd-weight 0.0 \
  run hands-rl/results_attribution seq ppo_rnd 1 150000 rnd000 --rnd-weight 0.0 \
  run hands-rl/results_attribution seq sac 1 150000 sacent4 --sac-target-entropy -4 \
  run hands-rl/results_attribution free trpo 2 150000 trpolowstd --trpo-log-std -1.5 --checkpoint-every 10000 --eval-checkpoints 10000,50000,150000 \
  run hands-rl/results_attribution free trpo 3 150000 trpolowstd --trpo-log-std -1.5 --checkpoint-every 10000 --eval-checkpoints 10000,50000,150000 \
  > /tmp/w_a.log 2>&1 &
echo "A=$!"
# worker B: attribution lowstd seq seeds + seed-45 extension (18 runs)
nohup bash /tmp/w_a.sh \
  run hands-rl/results_attribution seq trpo 2 150000 trpolowstd --trpo-log-std -1.5 --checkpoint-every 10000 --eval-checkpoints 10000,50000,150000 \
  run hands-rl/results_attribution seq trpo 3 150000 trpolowstd --trpo-log-std -1.5 --checkpoint-every 10000 --eval-checkpoints 10000,50000,150000 \
  run hands-rl/results free ppo_kl 4 150000 s45 \
  run hands-rl/results free ppo_kl 5 150000 s45 \
  run hands-rl/results free ppo_rb 4 150000 s45 \
  run hands-rl/results free ppo_rb 5 150000 s45 \
  run hands-rl/results free ppo_espo 4 150000 s45 \
  run hands-rl/results free ppo_espo 5 150000 s45 \
  run hands-rl/results free ppo_exo 4 150000 s45 \
  run hands-rl/results free ppo_exo 5 150000 s45 \
  run hands-rl/results seq ppo_kl 4 150000 s45 \
  run hands-rl/results seq ppo_kl 5 150000 s45 \
  run hands-rl/results seq ppo_rb 4 150000 s45 \
  run hands-rl/results seq ppo_rb 5 150000 s45 \
  run hands-rl/results seq ppo_espo 4 150000 s45 \
  run hands-rl/results seq ppo_espo 5 150000 s45 \
  run hands-rl/results seq ppo_exo 4 150000 s45 \
  run hands-rl/results seq ppo_exo 5 150000 s45 \
  > /tmp/w_b.log 2>&1 &
echo "B=$!"
sleep 3; tail -2 /tmp/w_a.log /tmp/w_b.log
"""


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PWD, timeout=20)
    # upload updated sources
    sftp = cli.open_sftp()
    for local, remote in [
        ("hands-rl/scripts/run_matrix.py", f"{BASE}/hands-rl/scripts/run_matrix.py"),
        ("hands-rl/algorithms/baselines.py", f"{BASE}/hands-rl/algorithms/baselines.py"),
    ]:
        sftp.put(local, remote)
        print("uploaded", local)
    script = LAUNCH.replace("__WORKER__", WORKER)
    _, out, err = cli.exec_command(script, timeout=120)
    print((out.read() + err.read()).decode(errors="replace"))
    cli.close()


if __name__ == "__main__":
    main()
