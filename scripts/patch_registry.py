"""Extend algorithm registries with TRPO and PPO-KL across the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EDITS = {
    "algorithms/__init__.py": [
        ('from .baselines import PPO_HPARAMS, SAC_HPARAMS, TD3_HPARAMS, make_ppo, make_sac, make_td3',
         'from .baselines import (PPO_HPARAMS, SAC_HPARAMS, TD3_HPARAMS, TRPO_HPARAMS,\n                         make_ppo, make_sac, make_td3, make_trpo)'),
        ('    "ppo":      ("on-policy", "PPO (Schulman et al., 2017)"),',
         '    "ppo":      ("on-policy", "PPO (Schulman et al., 2017)"),\n'
         '    "trpo":     ("on-policy", "TRPO (Schulman et al., 2015; CG + line search)"),\n'
         '    "ppo_kl":   ("on-policy", "PPO adaptive-KL-penalty variant (PPO-1)"),'),
        ('__all__ = ["ALGORITHMS", "PPOX", "make_ppo", "make_td3", "make_sac", "make_ppox",\n'
         '           "PBRSWrapper", "RNDRewardWrapper", "RNDModule",\n'
         '           "PPO_HPARAMS", "TD3_HPARAMS", "SAC_HPARAMS"]',
         '__all__ = ["ALGORITHMS", "PPOX", "make_ppo", "make_td3", "make_sac", "make_trpo",\n'
         '           "make_ppox", "PBRSWrapper", "RNDRewardWrapper", "RNDModule",\n'
         '           "PPO_HPARAMS", "TD3_HPARAMS", "SAC_HPARAMS", "TRPO_HPARAMS"]'),
    ],
    "scripts/aggregate.py": [
        ('ALGO_ORDER = ["ppo", "td3", "sac", "ppo_rb", "ppo_espo", "ppo_ppg",\n              "ppo_exo", "ppo_pbrs", "ppo_rnd"]',
         'ALGO_ORDER = ["ppo", "trpo", "ppo_kl", "td3", "sac", "ppo_rb", "ppo_espo",\n              "ppo_ppg", "ppo_exo", "ppo_pbrs", "ppo_rnd"]'),
        ('    "ppo": "PPO", "td3": "TD3", "sac": "SAC",',
         '    "ppo": "PPO", "trpo": "TRPO", "ppo_kl": "PPO-KL", "td3": "TD3", "sac": "SAC",'),
    ],
    "scripts/make_figures.py": [
        ('ALGO_ORDER = ["ppo", "td3", "sac", "ppo_rb", "ppo_espo", "ppo_ppg",\n               "ppo_exo", "ppo_pbrs", "ppo_rnd"]',
         'ALGO_ORDER = ["ppo", "trpo", "ppo_kl", "td3", "sac", "ppo_rb", "ppo_espo",\n               "ppo_ppg", "ppo_exo", "ppo_pbrs", "ppo_rnd"]'),
        ('LABEL = {"ppo": "PPO", "td3": "TD3", "sac": "SAC", "ppo_rb": "TR-PPO-RB",',
         'LABEL = {"ppo": "PPO", "trpo": "TRPO", "ppo_kl": "PPO-KL", "td3": "TD3", "sac": "SAC", "ppo_rb": "TR-PPO-RB",'),
        ('COLOR = {"ppo": "#075f57",',
         'COLOR = {"ppo": "#075f57", "trpo": "#7a4f9e", "ppo_kl": "#4a7d2f",'),
    ],
    "scripts/gen_readme.py": [
        ('LABEL = {"ppo": "PPO", "td3": "TD3", "sac": "SAC", "ppo_rb": "TR-PPO-RB",',
         'LABEL = {"ppo": "PPO", "trpo": "TRPO", "ppo_kl": "PPO-KL", "td3": "TD3", "sac": "SAC", "ppo_rb": "TR-PPO-RB",'),
    ],
    "scripts/gen_results_snippet.py": [
        ('LABEL = {"ppo": "PPO", "td3": "TD3", "sac": "SAC", "ppo_rb": "TR-PPO-RB",',
         'LABEL = {"ppo": "PPO", "trpo": "TRPO", "ppo_kl": "PPO-KL", "td3": "TD3", "sac": "SAC", "ppo_rb": "TR-PPO-RB",'),
    ],
    "scripts/gen_site.py": [
        ('    label = {"ppo": "PPO", "td3": "TD3", "sac": "SAC", "ppo_rb": "TR-PPO-RB",',
         '    label = {"ppo": "PPO", "trpo": "TRPO", "ppo_kl": "PPO-KL", "td3": "TD3", "sac": "SAC", "ppo_rb": "TR-PPO-RB",'),
    ],
}


def main():
    for rel, pairs in EDITS.items():
        path = ROOT / rel
        s = path.read_text(encoding="utf-8")
        for old, new in pairs:
            if old not in s:
                print(f"MISS in {rel}: {old[:60]!r}")
                continue
            s = s.replace(old, new, 1)
        path.write_text(s, encoding="utf-8", newline="\n")
        print("patched", rel)


if __name__ == "__main__":
    main()
