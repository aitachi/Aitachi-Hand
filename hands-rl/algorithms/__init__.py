"""Algorithm registry for the Aitachi-Hand comparison study."""
from .baselines import (PPO_HPARAMS, SAC_HPARAMS, TD3_HPARAMS, TRPO_HPARAMS,
                         make_ppo, make_sac, make_td3, make_trpo)
from .ppo_x import PPOX, make_ppox
from .reward_addons import PBRSWrapper, RNDRewardWrapper, RNDModule

ALGORITHMS = {
    # key          family            paper
    "ppo":      ("on-policy", "PPO (Schulman et al., 2017)"),
    "trpo":     ("on-policy", "TRPO (Schulman et al., 2015; CG + line search)"),
    "ppo_kl":   ("on-policy", "PPO adaptive-KL-penalty variant (PPO-1)"),
    "td3":      ("off-policy", "TD3 (Fujimoto et al., 2018)"),
    "sac":      ("off-policy", "SAC (Haarnoja et al., 2018)"),
    "ppo_rb":   ("ppo-improved", "TR-PPO-RB rollback (Truly-PPO)"),
    "ppo_espo": ("ppo-improved", "ESPO early stopping"),
    "ppo_ppg":  ("ppo-improved", "PPG phasic updates + auxiliary phase"),
    "ppo_exo":  ("ppo-improved", "ExO-PPO smooth off-policy surrogate"),
    "ppo_pbrs": ("reward-exploration", "PBRS potential shaping (Ng et al., 1999)"),
    "ppo_rnd":  ("reward-exploration", "RND intrinsic reward (Burda et al., 2018)"),
}

__all__ = ["ALGORITHMS", "PPOX", "make_ppo", "make_td3", "make_sac", "make_trpo",
           "make_ppox", "PBRSWrapper", "RNDRewardWrapper", "RNDModule",
           "PPO_HPARAMS", "TD3_HPARAMS", "SAC_HPARAMS", "TRPO_HPARAMS"]
