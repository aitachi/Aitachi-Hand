"""One-shot remote survey over password SSH (paramiko)."""
import sys

import paramiko

HOST, PORT, USER, PWD = "connect.nmb2.seetacloud.com", 17281, "root", "zC2ydRyLCjK9"

CMDS = [
    "hostname; uname -m",
    "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader",
    "df -h / /root/autodl-tmp 2>/dev/null | tail -2",
    "python3 --version; which pip3",
    "python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>/dev/null || echo no-torch",
    "free -g | head -2 | tail -1; nproc",
    "ls /root/autodl-tmp 2>/dev/null | head -3",
]


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PWD, timeout=20)
    for cmd in CMDS:
        _, out, err = cli.exec_command(cmd, timeout=60)
        print("$", cmd)
        print((out.read() + err.read()).decode(errors="replace").strip()[:500])
    cli.close()


if __name__ == "__main__":
    main()
