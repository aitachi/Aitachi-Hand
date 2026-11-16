"""Generate the four-language README (experiment report) from summary.json.

Languages: English | 中文 | 日本語 | 한국어
Run after scripts/aggregate.py whenever results change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL = {"ppo": "PPO", "trpo": "TRPO", "ppo_kl": "PPO-KL", "td3": "TD3", "sac": "SAC", "ppo_rb": "TR-PPO-RB",
         "ppo_espo": "ESPO", "ppo_ppg": "PPG", "ppo_exo": "ExO-PPO",
         "ppo_pbrs": "PBRS+PPO", "ppo_rnd": "RND+PPO"}
ORDER = list(LABEL)


def fmt(v, d=3, dash="--"):
    if v is None:
        return dash
    try:
        return dash if v != v else f"{v:.{d}f}"
    except TypeError:
        return dash


def results_table(summary, task, L):
    rows = []
    best = None
    for algo in ORDER:
        e = next((e for e in summary if e["task"] == task and e["algo"] == algo), None)
        if not e:
            continue
        score = (e["success_rate"], -(e.get("finish_time") or 9e9))
        if best is None or score > best[0]:
            best = (score, LABEL[algo])
    for algo in ORDER:
        e = next((e for e in summary if e["task"] == task and e["algo"] == algo), None)
        if not e:
            continue
        name = LABEL[algo] + (" ⭐" if best and best[1] == LABEL[algo] else "")
        rows.append(
            f'| {name} | {e["seeds"]} | {fmt(e["success_rate"])}±{fmt(e.get("success_rate_std"))} '
            f'| {fmt(e.get("finish_time"), 2)} | {fmt(e.get("penetration_mean_mm"), 2)} '
            f'| {fmt(e.get("train_auc"), 2)} | {int(e.get("wall_seconds", 0) / 60)} |')
    head = (f'| {L["m_method"]} | {L["m_seeds"]} | {L["m_success"]} | {L["m_time"]} '
            f'| {L["m_pen"]} | {L["m_auc"]} | {L["m_wall"]} |')
    sep = "|---|---|---|---|---|---|---|"
    return "\n".join([head, sep] + rows), (best[1] if best else "--")


LANGS = {
    "en": {
        "nav": "Language: **English** | [中文](README.zh.md) | [日本語](README.ja.md) | [한국어](README.ko.md)",
        "title": "# Aitachi-Hands — Open 17-DOF Dexterous-Hand Platform & PPO-Improvement Controlled Study",
        "intro": (
            "**Aitachi-Hands** couples the CC-BY-4.0 [ORCA Hand v2](https://github.com/orcahand) "
            "(17 DOF, tendon-driven) with measured Feetech-HLS servo calibration, a safety-gated "
            "serial protocol, MuJoCo/Isaac-Lab simulation, and a controlled reinforcement-learning "
            "comparison — everything from one repository."),
        "quick": "## Quick Start",
        "m_method": "Method", "m_seeds": "Seeds", "m_success": "Success ±σ",
        "m_time": "Time (s)", "m_pen": "Pen. (mm)", "m_auc": "Reward AUC",
        "m_wall": "Wall (min)",
        "t_free": "### Task 1 · Collision-free free posing (5 fingers, no self-fighting)",
        "t_seq": "### Task 2 · Rapid gesture sequence V → OK → OPEN",
        "best": "Best (success, then time)",
        "protocol_head": "## Experimental Protocol",
        "protocol": (
            "- **Matrix**: 2 tasks × 11 methods × 3 seeds = 66 runs, 150 000 environment steps each.\n"
            "- **Shared backbone**: residual action `cmd = prior + 0.3·π(s)` with 0.15 rad/cycle rate limit; "
            "the zero policy is a solvable direct baseline, so all methods optimize the same correct prior.\n"
            "- **Networks**: 256×256 for every method (PPG uses a shared trunk by construction).\n"
            "- **Evaluation**: deterministic policy, 25 fixed seeds (7001–7025) per run; success, arrival/"
            "sequence time and self-penetration are hard metrics.\n"
            "- **Stack**: MuJoCo 3.3.7 · torch 2.11.0+cu130 · Stable-Baselines3 2.9.2 · RTX 3070 Laptop.\n"
            "- **Training curves use episode reward**: sampled-noise rollouts structurally record ~0 "
            "training-time successes (hold-window criterion), a known artifact documented in the report."),
        "deviations_head": "## Disclosed Deviations from Original Papers",
        "deviations": (
            "1. **TR-PPO-RB**: per-state KL is proxied by |log ρ| > 0.5 (SB3 does not store old distribution parameters); "
            "rollback coefficient c = (1+ε)/(1−ε) = 1.5.\n"
            "2. **RND**: single value head (paper uses separate value heads + non-episodic intrinsic returns); "
            "intrinsic weight 0.5 with running normalization.\n"
            "3. **ExO-PPO**: smooth compression implemented as ε·tanh saturation of log-ratio beyond the trust region.\n"
            "4. **OPPO**: cited only (LLM serving-overlap systems paper; no inference loop in this robot study).\n"
            "5. PBRS shaping adds no terminal-potential correction on time-limit truncation (success terminates at Φ≈0).\n"
            "6. **TD3/SAC cadence**: gradient updates every 4 env steps in batches of 4 (update-to-data ratio 1:1, "
            "identical to the papers; the batching is a wall-clock optimization on a single laptop GPU).\n"
            "7. **TD3/SAC device**: mixed by measurement — free TD3 s2/s3 and SAC s1/s3+seq ran on GPU; SAC s2 ran on CPU (WSL thread contention made CPU slower: 3h+ vs 2h04m GPU). Updates are mathematically equivalent; devices are recorded per-run in config.json.\n"
            "8. **RND module device**: forced to CPU after per-child CUDA contexts deadlocked inside SubprocVecEnv workers (two consecutive seq hangs); free-task RND runs completed on the GPU module before the fix — values are numerically equivalent.\n"
            "9. **RND vec-env backend**: after four consecutive seq-task pipe deadlocks, RND runs use DummyVecEnv (8 envs stepped in one process). Rollout semantics (n_steps per env, batch sizes) are bit-identical; only stepping is sequential."),
        "repro_head": "## Reproduce",
        "assets": "> The 111 MB MuJoCo model is not committed — run `python scripts/fetch_assets.py` "
                  "(or set `AHAND_MODEL_ROOT` to an existing orca_v2 copy).",
        "limits_head": "## Limitations",
        "limits": (
            "Simulation-scoped results; deployment requires the separate guarded replay pipeline "
            "(rate-limited segments, current/temperature/voltage/stall protection, camera correction). "
            "One fixed budget — per-method tuning could reorder rankings; this is a *controlled* comparison."),
        "results_head": "## Results",
        "cross_head": "Cross-study context (RLP benchmark)",
        "c_method": "Method", "c_note": "Note",
        "c_src": "Source",
        "links_head": "## Deliverables",
        "links": (
            "- 📕 Textbook (19-page PDF, Chinese): `textbook/main.pdf`\n"
            "- 📄 IEEE paper: `paper/ieee_main.pdf` (results auto-generated from `results/summary.json`)\n"
            "- 📊 12 comparison figures: `figures/`\n"
            "- 🧾 Servo command appendix: textbook Appendix A / `docs/`\n"
            "- 🌐 Project site: http://115.190.62.87/Aitachi-Hand/"),
    },
    "zh": {
        "nav": "语言：[English](README.md) | **中文** | [日本語](README.ja.md) | [한국어](README.ko.md)",
        "title": "# Aitachi-Hands —— 开源 17 轴灵巧手平台与 PPO 改进方法受控对比研究",
        "intro": (
            "**Aitachi-Hands** 是 17 自由度腱绳驱动灵巧手平台，机械本体采用开源 CC BY 4.0 手体设计 [ORCA Hand v2](https://github.com/orcahand)（依许可保留署名），"
            "打通实测飞特 HLS 舵机标定、带安全护栏的串口协议、MuJoCo / Isaac Lab 仿真，"
            "以及一套受控强化学习对比实验——全部出自同一仓库。"),
        "quick": "## 快速开始",
        "m_method": "方法", "m_seeds": "种子数", "m_success": "成功率 ±σ",
        "m_time": "时间 (s)", "m_pen": "穿透 (mm)", "m_auc": "奖励AUC",
        "m_wall": "耗时 (分)",
        "t_free": "### 任务一 · 五指自由活动不打架（自碰撞约束达标）",
        "t_seq": "### 任务二 · V → OK → OPEN 快速手势序列",
        "best": "最优（先成功率后时间）",
        "protocol_head": "## 实验协议（严格口径）",
        "protocol": (
            "- **矩阵**：2 任务 × 11 方法 × 3 种子 = 66 跑，每跑 150 000 环境步。\n"
            "- **共享骨干**：残差动作 `指令 = 目标先验 + 0.3·π(s)`，每周期限速 0.15 rad；"
            "零策略即可解直接基线，所有方法在同一正确先验之上优化。\n"
            "- **网络**：全部 256×256（PPG 按原论文用共享主干）。\n"
            "- **评估**：确定性策略 + 25 个固定种子（7001–7025）；成功率 / 到位或序列时间 / 自穿透为硬指标。\n"
            "- **环境**：MuJoCo 3.3.7 · torch 2.11.0+cu130 · Stable-Baselines3 2.9.2 · RTX 3070 Laptop。\n"
            "- **训练曲线采用回合奖励**：采样噪声使训练期成功率结构性≈0（保持窗判据被噪声打破），"
            "该统计假象在报告中如实披露。"),
        "deviations_head": "## 与原论文的实现偏差（如实披露）",
        "deviations": (
            "1. **TR-PPO-RB**：逐状态 KL 以 |log ρ|>0.5 代理（SB3 不保存旧分布参数）；rollback 系数 c=(1+ε)/(1−ε)=1.5。\n"
            "2. **RND**：单价值头简化（原文双价值头+非回合内在回报）；内在权重 0.5 + 滑动归一化。\n"
            "3. **ExO-PPO**：以 ε·tanh 软饱和实现越界平滑压缩。\n"
            "4. **OPPO**：仅引用不实跑（LLM 服务重叠调度的系统论文，本机器人研究无推理服务环）。\n"
            "5. PBRS 超时截断不加终端势修正（成功终止时 Φ≈0）。\n"
            "6. **TD3/SAC 更新节奏**：每 4 环境步批量更新 4 次（更新/数据比仍为 1:1，与原论文一致；批量化仅为单笔记本 GPU 的墙钟优化）。\n"
            "7. **TD3/SAC 设备**：按实测混合——free TD3 s2/s3 与 SAC s1/s3+seq 用 GPU；SAC s2 用 CPU（WSL 线程争用使 CPU 更慢：3h+ vs GPU 2h04m）。更新数学等价，逐跑设备记录于 config.json。\n"
            "8. **RND 模块设备**：SubprocVecEnv 子进程各自的 CUDA 上下文死锁（seq 连续两次挂死）后强制改 CPU；free 任务 RND 在修复前以 GPU 模块完成——数值等价。\n"
            "9. **RND 向量环境后端**：seq 任务连续四次管道死锁后，RND 跑改用 DummyVecEnv（8 环境单进程串行步进）。rollout 语义（每环境 n_steps、批大小）逐位一致，仅执行串行化。"),
        "repro_head": "## 一键复现",
        "assets": "> 111 MB MuJoCo 模型不入仓——先跑 `python scripts/fetch_assets.py`"
                  "（或将 `AHAND_MODEL_ROOT` 指向已有 orca_v2 目录）。",
        "limits_head": "## 局限",
        "limits": (
            "结果限于仿真；真机部署需另走受护栏的分段回放管线（限速/电流/温度/电压/停滞四重保护+摄像头矫正）。"
            "本对比固定单一预算——逐方法调参可能改变排名，这是一次*受控*对比而非穷举对比。"),
        "results_head": "## 实验结果",
        "cross_head": "跨研究对照（RLP 基准）",
        "c_method": "方法", "c_note": "注记",
        "c_src": "来源",
        "links_head": "## 交付物",
        "links": (
            "- 📕 全栈教材（19 页 PDF，中文）：`textbook/main.pdf`\n"
            "- 📄 IEEE 论文：`paper/ieee_main.pdf`（结果表由 `results/summary.json` 自动生成）\n"
            "- 📊 12 张对比学术图：`figures/`\n"
            "- 🧾 舵机指令附录：教材附录 A / `docs/`\n"
            "- 🌐 项目站点：http://115.190.62.87/Aitachi-Hand/"),
    },
    "ja": {
        "nav": "言語：[English](README.md) | [中文](README.zh.md) | **日本語** | [한국어](README.ko.md)",
        "title": "# Aitachi-Hands — オープンソース17自由度デキスタラスハンドとPPO改良法の制御比較研究",
        "intro": (
            "**Aitachi-Hands** は CC-BY-4.0 の [ORCA Hand v2](https://github.com/orcahand)"
            "（17自由度・腱駆動）を本体とし、実測された Feetech HLS サーボ校正、"
            "安全保護付きシリアルプロトコル、MuJoCo / Isaac Lab シミュレーション、"
            "そして制御された強化学習比較を一つのリポジトリで提供します。"),
        "quick": "## クイックスタート",
        "m_method": "手法", "m_seeds": "シード", "m_success": "成功率 ±σ",
        "m_time": "時間 (s)", "m_pen": "貫通 (mm)", "m_auc": "報酬AUC",
        "m_wall": "学習時間 (分)",
        "t_free": "### タスク1 · 自由姿勢（5指・自己衝突なし）",
        "t_seq": "### タスク2 · V → OK → OPEN 高速ジェスチャー連鎖",
        "best": "最良（成功率→時間）",
        "protocol_head": "## 実験プロトコル",
        "protocol": (
            "- **行列**: 2タスク × 11手法 × 3シード = 66実行、各150,000環境ステップ。\n"
            "- **共通バックボーン**: 残差アクション `cmd = prior + 0.3·π(s)`（0.15 rad/サイクル速度制限）。\n"
            "- **ネットワーク**: 全手法 256×256（PPG は共有トランク）。\n"
            "- **評価**: 決定論的ポリシー、25固定シード（7001–7025）。\n"
            "- **スタック**: MuJoCo 3.3.7 · torch 2.11.0+cu130 · SB3 2.9.2 · RTX 3070 Laptop。"),
        "deviations_head": "## 原論文からの相違（開示）",
        "deviations": (
            "1. **TR-PPO-RB**: 状態ごとのKLを |log ρ|>0.5 で代理。c=1.5。\n"
            "2. **RND**: 単一価値ヘッド簡略版、重み0.5。\n"
            "3. **ExO-PPO**: ε·tanh 軟飽和による平滑圧縮。\n"
            "4. **OPPO**: 引用のみ（LLMサービス論文のため実行せず）。\n"
            "5. PBRS はタイムアウト終端でポテンシャル補正なし。\n"
            "6. **TD3/SAC**: 4ステップごとに4更新（UTD 1:1 は原論文同様）。"),
        "repro_head": "## 再現手順",
        "assets": "> 111 MB の MuJoCo モデルはリポジトリ外 — `python scripts/fetch_assets.py` を実行"
                  "（または `AHAND_MODEL_ROOT` を既存の orca_v2 に設定）。",
        "limits_head": "## 限界",
        "limits": (
            "結果はシミュレーション範囲。実機展開には保護付き再生パイプラインが別途必要。"
            "単一予算の制御比較であり、手法別チューニングで順位は変わり得る。"),
        "results_head": "## 結果",
        "cross_head": "他研究との对照（RLP ベンチマーク）",
        "c_method": "手法", "c_note": "備考",
        "c_src": "出典",
        "links_head": "## 成果物",
        "links": (
            "- 📕 教科書（19ページPDF・中国語）: `textbook/main.pdf`\n"
            "- 📄 IEEE論文: `paper/ieee_main.pdf`\n"
            "- 📊 12枚の比較図: `figures/`\n"
            "- 🧾 サーボ指令付録: 教科書付録A\n"
            "- 🌐 プロジェクトサイト: http://115.190.62.87/Aitachi-Hand/"),
    },
    "ko": {
        "nav": "언어: [English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | **한국어**",
        "title": "# Aitachi-Hands — 오픈소스 17-자유도 로봇 손 플랫폼과 PPO 개선법 통제 비교 연구",
        "intro": (
            "**Aitachi-Hands**는 CC-BY-4.0 [ORCA Hand v2](https://github.com/orcahand)"
            "(17자유도, 텐던 구동) 본체에 실측된 Feetech HLS 서보 캘리브레이션, "
            "안전 보호 직렬 프로토콜, MuJoCo / Isaac Lab 시뮬레이션, "
            "그리고 통제된 강화학습 비교를 하나의 저장소로 제공합니다."),
        "quick": "## 빠른 시작",
        "m_method": "방법", "m_seeds": "시드", "m_success": "성공률 ±σ",
        "m_time": "시간 (s)", "m_pen": "관통 (mm)", "m_auc": "보상AUC",
        "m_wall": "학습시간 (분)",
        "t_free": "### 과제 1 · 자유 자세(5지, 자가충돌 없음)",
        "t_seq": "### 과제 2 · V → OK → OPEN 빠른 제스처 연쇄",
        "best": "최우수(성공률→시간)",
        "protocol_head": "## 실험 프로토콜",
        "protocol": (
            "- **행렬**: 2과제 × 11방법 × 3시드 = 66실행, 각 150,000 환경 스텝.\n"
            "- **공통 백본**: 잔차 행동 `cmd = prior + 0.3·π(s)`(주기당 0.15 rad 속도 제한).\n"
            "- **네트워크**: 전 방법 256×256(PPG는 공유 트렁크).\n"
            "- **평가**: 결정론 정책, 25 고정 시드(7001–7025).\n"
            "- **스택**: MuJoCo 3.3.7 · torch 2.11.0+cu130 · SB3 2.9.2 · RTX 3070 Laptop."),
        "deviations_head": "## 원 논문과의 차이(공개)",
        "deviations": (
            "1. **TR-PPO-RB**: 상태별 KL을 |log ρ|>0.5로 대리, c=1.5.\n"
            "2. **RND**: 단일 가치 헤드 간소화, 가중치 0.5.\n"
            "3. **ExO-PPO**: ε·tanh 연화 포화로 평활 압축.\n"
            "4. **OPPO**: 인용만(LLM 서빙 시스템 논문).\n"
            "5. PBRS 타임아웃 종단 전위 보정 없음.\n"
            "6. **TD3/SAC**: 4스텝당 4더 업데이트(UTD 1:1 동일)."),
        "repro_head": "## 재현",
        "assets": "> 111 MB MuJoCo 모델은 저장소에 포함되지 않음 — `python scripts/fetch_assets.py` 실행"
                  "(또는 `AHAND_MODEL_ROOT`를 기존 orca_v2로 설정).",
        "limits_head": "## 한계",
        "limits": (
            "결과는 시뮬레이션 범위. 실기기 배포에는 보호된 재생 파이프라인이 별도로 필요. "
            "단일 예산의 통제 비교이며 방법별 튜닝으로 순위는 변할 수 있음."),
        "results_head": "## 결과",
        "cross_head": "교차 연구 대조(RLP 벤치마크)",
        "c_method": "방법", "c_note": "비고",
        "c_src": "출처",
        "links_head": "## 산출물",
        "links": (
            "- 📕 교재(19쪽 PDF, 중국어): `textbook/main.pdf`\n"
            "- 📄 IEEE 논문: `paper/ieee_main.pdf`\n"
            "- 📊 12장 비교 그림: `figures/`\n"
            "- 🧾 서보 명령 부록: 교재 부록A\n"
            "- 🌐 프로젝트 사이트: http://115.190.62.87/Aitachi-Hand/"),
    },
}

REPRO = {
    "en": """```bash
git clone https://github.com/aitachi/Aitachi-Hand && cd Aitachi-Hands
python scripts/fetch_assets.py          # official orca_v2 model (not committed)
bash scripts/smoke_all.sh               # 13-path smoke (~4 min)
bash scripts/run_matrix_all.sh          # full 66-run matrix (~6-8 h, RTX 3070)
python scripts/aggregate.py             # results/summary.{csv,json}
python scripts/make_figures.py          # figures/fig01..fig12
python scripts/gen_results_snippet.py   # refresh IEEE paper numbers
python scripts/gen_readme.py            # refresh this README
```""",
    "zh": """```bash
git clone https://github.com/aitachi/Aitachi-Hand && cd Aitachi-Hands
python scripts/fetch_assets.py          # 官方 orca_v2 模型（不入仓）
bash scripts/smoke_all.sh               # 13 条路径冒烟（约 4 分钟）
bash scripts/run_matrix_all.sh（约 6-8 小时，RTX 3070）
python scripts/aggregate.py             # results/summary.{csv,json}
python scripts/make_figures.py          # figures/fig01..fig12
python scripts/gen_results_snippet.py   # 刷新 IEEE 论文数字
python scripts/gen_readme.py            # 刷新本 README
```""",
    "ja": """```bash
git clone https://github.com/aitachi/Aitachi-Hand && cd Aitachi-Hands
python scripts/fetch_assets.py          # 公式 orca_v2 モデル
bash scripts/smoke_all.sh               # 13パス スモークテスト(~4分)
bash scripts/run_matrix_all.sh          # 66実行の全行列(~6-8時間)
python scripts/aggregate.py             # results/summary.{csv,json}
python scripts/make_figures.py          # figures/fig01..fig12
```""",
    "ko": """```bash
git clone https://github.com/aitachi/Aitachi-Hand && cd Aitachi-Hands
python scripts/fetch_assets.py          # 공식 orca_v2 모델
bash scripts/smoke_all.sh               # 13경로 스모크(~4분)
bash scripts/run_matrix_all.sh          # 66실행 전체 매트릭스(~6-8시간)
python scripts/aggregate.py             # results/summary.{csv,json}
python scripts/make_figures.py          # figures/fig01..fig12
```""",
}


def render_cross(L):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cross_study import RLP_CROSS_STUDY, SOURCE_NOTE
    head = f"| {L['c_method']} | E1 | E2 | E3 | {L['c_note']} |"
    rows = [head, "|---|---|---|---|---|"]
    for m, e1, e2, e3, note in RLP_CROSS_STUDY:
        rows.append(f"| {m} | {e1} | {e2} | {e3} | {note} |")
    return "\n".join([
        f"## {L['cross_head']}",
        "",
        "\n".join(rows),
        "",
        f"*{L['c_src']}: {SOURCE_NOTE.split('：', 1)[-1]}*",
        "",
    ])


def _g(L, key, en):
    """Localized string with English fallback (ja/ko reuse EN for table headers)."""
    return L.get(key, en)


def render_diag_table(summary, L):
    """Optimizer diagnostics for the PPO family (training tail)."""
    h_method = _g(L, "m_method", "Method")
    head = (f"| {h_method} | clip frac | approx KL | max ratio | ESPO stop% | PPG aux KL |")
    rows = [head, "|---|---|---|---|---|---|"]
    ppo_family = ["ppo", "trpo", "ppo_kl", "ppo_rb", "ppo_espo", "ppo_ppg",
                  "ppo_exo", "ppo_pbrs", "ppo_rnd"]
    for algo in ppo_family:
        e = next((e for e in summary if e["algo"] == algo and e["task"] == "seq"), None)
        if not e:
            continue
        def f(k, d=4):
            v = e.get(k)
            return "--" if v is None or v != v else f"{v:.{d}f}"
        stop = f("espo_stop_frac", 2)
        aux = f("ppg_aux_kl", 4)
        rows.append(f"| {LABEL[algo]} | {f('clip_frac')} | {f('approx_kl')} | {f('ratio_max', 2)} | {stop} | {aux} |")
    return "\n".join(rows)


def render_eff_table(summary, L):
    """Sample efficiency + wall clock, both tasks."""
    h_method = _g(L, "m_method", "Method")
    head = (f"| {h_method} | free: steps→90% | free AUC | free wall | seq: steps→90% | seq AUC | seq wall |")
    rows = [head, "|---|---|---|---|---|---|---|"]
    for algo in ORDER:
        cells = [LABEL[algo]]
        for task in ("free", "seq"):
            e = next((e for e in summary if e["algo"] == algo and e["task"] == task), None)
            if not e:
                cells += ["--", "--", "--"]
                continue
            s90 = e.get("steps_to_90")
            s90 = f"{int(s90/1000)}k" if s90 else "--"
            auc = e.get("train_auc")
            auc = f"{auc:.2f}" if auc is not None and auc == auc else "--"
            wall = f"{int((e.get('wall_seconds') or 0)/60)}m"
            cells += [s90, auc, wall]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render_phase_table(summary, L):
    """Task-2 per-phase decomposition (successful episodes)."""
    h_method = _g(L, "m_method", "Method")
    head = f"| {h_method} | V (s) | OK (s) | OPEN (s) | total (s) |"
    rows = [head, "|---|---|---|---|---|"]
    for algo in ORDER:
        e = next((e for e in summary if e["algo"] == algo and e["task"] == "seq"), None)
        if not e or e.get("phase_v") is None:
            continue
        v, ok, op = e["phase_v"], e["phase_ok"], e["phase_open"]
        rows.append(f"| {LABEL[algo]} | {v:.2f} | {ok:.2f} | {op:.2f} | {v+ok+op:.2f} |")
    return "\n".join(rows)


FIGURES = [
    ("fig01_learning_free.png", "Task 1 learning curves (smoothed episode reward, 3 seeds)"),
    ("fig02_learning_seq.png", "Task 2 learning curves"),
    ("fig03_final_success.png", "Final success rate (deterministic 25-seed eval, mean±σ)"),
    ("fig04_finish_time.png", "Arrival / sequence time"),
    ("fig05_sample_efficiency.png", "Sample efficiency: steps to 90% of final training reward"),
    ("fig06_ppo_diagnostics.png", "PPO-family optimizer diagnostics (clip/KL/ratio/AUC)"),
    ("fig07_espo_stopping.png", "ESPO early stopping: KL trace + stop-epoch histogram"),
    ("fig08_reward_group.png", "Reward-shaping group (PBRS/RND vs PPO) + RND intrinsic tail"),
    ("fig09_eval_box.png", "Evaluation time distributions (box plots)"),
    ("fig10_phase_decomposition.png", "Task-2 per-phase time decomposition (V/OK/OPEN)"),
    ("fig11_wallclock.png", "Wall-clock efficiency: final success vs training time"),
    ("fig12_safety_penetration.png", "Safety proxy: max self-penetration during evaluation"),
    ("fig13_cross_study.png", "Cross-study comparison (RLP benchmark, static data)"),
]

FIG_CAPTIONS_ZH = {
    "fig01_learning_free.png": "任务一学习曲线（平滑回合奖励，3 种子）",
    "fig02_learning_seq.png": "任务二学习曲线",
    "fig03_final_success.png": "最终成功率（确定性 25 种子评估，均值±σ）",
    "fig04_finish_time.png": "到位/序列时间对比",
    "fig05_sample_efficiency.png": "样本效率：达到最终训练奖励 90% 所需步数",
    "fig06_ppo_diagnostics.png": "PPO 族优化器诊断（clip 率/KL/最大比率/AUC）",
    "fig07_espo_stopping.png": "ESPO 早停：KL 轨迹与停止轮分布",
    "fig08_reward_group.png": "奖励塑形组（PBRS/RND 对比 PPO）与 RND 内在量尾部",
    "fig09_eval_box.png": "评估集时间分布（箱线图）",
    "fig10_phase_decomposition.png": "任务二相位时间分解（V/OK/OPEN）",
    "fig11_wallclock.png": "墙钟效率：最终成功率 vs 训练时长",
    "fig12_safety_penetration.png": "安全代理指标：评估期最大自穿透",
    "fig13_cross_study.png": "跨研究对比（RLP 基准，静态数据）",
}


def render_figures(lang):
    parts = []
    for fname, en_cap in FIGURES:
        cap = FIG_CAPTIONS_ZH.get(fname, en_cap) if lang == "zh" else en_cap
        parts.append(f"![{cap}](figures/{fname})")
        parts.append(f"*{cap}*")
        parts.append("")
    return "\n".join(parts)


def render(lang: str, summary, n_runs: int) -> str:
    L = LANGS[lang]
    zh = lang == "zh"
    out = [L["title"], "", L["intro"], "", L["nav"], ""]
    out += [L["results_head"] + f" ({n_runs} runs)", ""]
    for task in ("free", "seq"):
        table, best = results_table(summary, task, L)
        out += [L[f"t_{task}"], table, f'*{L["best"]}: **{best}***', ""]
    # T3 diagnostics
    out += [_g(L, "diag_head", "### Optimizer Diagnostics (PPO family, Task 2 training tail)" if not zh
               else "### 优化器诊断（PPO 族，任务二训练尾部）"), "", render_diag_table(summary, L), ""]
    # T4 efficiency
    out += [_g(L, "eff_head", "### Sample Efficiency & Wall Clock" if not zh
               else "### 样本效率与墙钟"), "", render_eff_table(summary, L), ""]
    # T5 phase decomposition
    out += [_g(L, "phase_head", "### Task-2 Phase Decomposition (successful episodes)" if not zh
               else "### 任务二相位分解（成功回合）"), "", render_phase_table(summary, L), ""]
    # figures gallery
    out += [_g(L, "figs_head", "### Comparison Figures (13)" if not zh
               else "### 对比图集（13 张）"), "", render_figures(lang)]
    out += [L["protocol_head"], L["protocol"], ""]
    out += [L["deviations_head"], L["deviations"], ""]
    out += [render_cross(L)]
    out += [L["quick"], REPRO[lang], "", L["assets"], ""]
    out += [L["limits_head"], L["limits"], ""]
    out += [L["links_head"], L["links"], ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="hands-rl/results")
    parser.add_argument("--out", default="README.md")
    args = parser.parse_args()
    data = json.loads((Path(args.results) / "summary.json").read_text(encoding="utf-8"))
    summary, n_runs = data["summary"], data["runs"]
    root = Path(args.out).parent
    header = ("<!-- AUTO-GENERATED by scripts/gen_readme.py - edit that script, "
              "not this file -->\n\n")
    for lang, fname in (("en", "README.md"), ("zh", "README.zh.md"),
                        ("ja", "README.ja.md"), ("ko", "README.ko.md")):
        (root / fname).write_text(header + render(lang, summary, n_runs) + "\n",
                                  encoding="utf-8")
        print(f"wrote {fname} ({n_runs} runs)")


if __name__ == "__main__":
    main()
