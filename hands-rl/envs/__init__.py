"""Aitachi-Hand RL environments (MuJoCo ORCA v2 right hand)."""
from gymnasium import register

from .free_task import AitachiFreeEnv
from .seq_task import AitachiSeqEnv

register(id="AitachiFree-v1", entry_point="envs.free_task:AitachiFreeEnv")
register(id="AitachiSeq-v1", entry_point="envs.seq_task:AitachiSeqEnv")

TASKS = {
    "free": "AitachiFree-v1",
    "seq": "AitachiSeq-v1",
}

__all__ = ["AitachiFreeEnv", "AitachiSeqEnv", "TASKS", "register"]
