"""Generate the static Aitachi-Hands project site (13 pages) into site/.

Pages: index + hardware / servo / software / drivers / mujoco / isaac /
rl_framework / algos16 / experiments / report / paper / resources.
Run: python scripts/gen_site.py  (stdlib only; reads results/summary.json)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
GITHUB = "https://github.com/aitachi/Aitachi-Hand"

CSS = """
:root{--ink:#16232a;--mut:#5a707a;--teal:#075f57;--teal2:#0e8a76;--wash:#eaf4f0;
--line:#d7e3e0;--bg:#f4f8f6;--warn:#a05018;--card:#ffffff}
*{box-sizing:border-box}body{margin:0;font:16px/1.75 "Segoe UI","Microsoft YaHei",system-ui,sans-serif;color:var(--ink);background:var(--bg)}
header.site{background:linear-gradient(135deg,#0b3a34,#0e5d50 60%,#11876f);color:#eefaf6;padding:46px 0 40px}
.wrap{width:min(1180px,calc(100% - 40px));margin:auto}
header.site h1{margin:.1em 0 .15em;font-size:clamp(1.9rem,4.5vw,3rem);letter-spacing:-.02em}
header.site p{margin:0;color:#cfeee4;max-width:860px}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.badges span{border:1px solid #3f8d7c;background:#0d4a40;border-radius:999px;padding:4px 13px;font-size:.85rem}
nav.top{background:#0a2f29;position:sticky;top:0;z-index:9}
nav.top a{display:inline-block;color:#bfe8dc;text-decoration:none;padding:10px 13px;font-size:.92rem;border-bottom:2px solid transparent}
nav.top a:hover,nav.top a.on{color:#fff;border-bottom-color:#35c9a3}
main{padding:30px 0 60px}
h2{color:var(--teal);font-size:1.5rem;margin:1.6em 0 .5em}
h3{color:var(--teal2);margin:1.4em 0 .4em}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:22px 24px;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;text-decoration:none;color:var(--ink);transition:.15s}
.tile:hover{border-color:var(--teal2);box-shadow:0 3px 14px #0a3c3216;transform:translateY(-2px)}
.tile b{display:block;color:var(--teal);font-size:1.04rem;margin-bottom:4px}
.tile span{font-size:.88rem;color:var(--mut)}
table{border-collapse:collapse;width:100%;font-size:.9rem;background:#fff;margin:.6em 0}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:var(--wash);color:var(--teal)}
tr:nth-child(even) td{background:#f8fbfa}
code,pre{font-family:"Cascadia Code",Consolas,monospace}
code{background:var(--wash);padding:1px 6px;border-radius:4px;font-size:.9em}
pre{background:#0d2b26;color:#d7efe7;padding:16px 18px;border-radius:10px;overflow:auto;font-size:.86rem;line-height:1.6}
pre b{color:#57d9b4;font-weight:600}
.note{border-left:4px solid var(--teal2);background:var(--wash);padding:12px 16px;border-radius:0 8px 8px 0}
.warn{border-left:4px solid var(--warn);background:#fdf3ea;padding:12px 16px;border-radius:0 8px 8px 0}
img.fig{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff;margin:8px 0}
.figcap{font-size:.83rem;color:var(--mut);margin:.2em 0 1.2em}
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:860px){.cols2{grid-template-columns:1fr}}
footer{border-top:1px solid var(--line);color:var(--mut);padding:22px 0 40px;font-size:.88rem}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.kpi div{background:#0d4a40;border-radius:10px;padding:14px;text-align:center;color:#fff}
.kpi b{display:block;font-size:1.7rem;color:#5fe0bd}
.kpi span{font-size:.82rem;color:#bfe8dc}
.pill{display:inline-block;background:var(--wash);color:var(--teal);border-radius:6px;padding:2px 9px;font-size:.8rem;font-weight:700;margin-right:6px}
a{color:var(--teal2)}
"""

NAV = [
    ("index.html", "首页"),
    ("hardware.html", "硬件设计与安装"),
    ("servo.html", "舵机指令集"),
    ("software.html", "软件对齐"),
    ("drivers.html", "驱动安装"),
    ("mujoco.html", "MuJoCo 教程"),
    ("isaac.html", "Isaac Lab 教程"),
    ("rl_framework.html", "RL 框架与双任务"),
    ("algos16.html", "16 算法分类"),
    ("experiments.html", "对比实验专区"),
    ("report.html", "实验报告"),
    ("paper.html", "SCI 论文"),
    ("resources.html", "资源下载"),
]


def page(name: str, title: str, body: str, on: str = "") -> str:
    nav = "".join(
        f'<a href="{href}"{" class=on" if href == on else ""}>{label}</a>'
        for href, label in NAV)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Aitachi-Hands</title><style>{CSS}</style></head><body>
<nav class="top"><div class="wrap">{nav}</div></nav>
{body}
<footer><div class="wrap">Aitachi-Hands · 2026 · <a href="{GITHUB}">GitHub</a> ·
教材/论文/图/数据全部随仓发布 · 站点由 <code>scripts/gen_site.py</code> 生成</div></footer>
</body></html>"""


def hero(title: str, sub: str) -> str:
    return (f'<header class="site"><div class="wrap"><h1>{title}</h1><p>{sub}</p></div></header>')


def wrap(inner: str) -> str:
    return f'<main><div class="wrap">{inner}</div></main>'


def results_tables(summary):
    label = {"ppo": "PPO", "trpo": "TRPO", "ppo_kl": "PPO-KL", "td3": "TD3", "sac": "SAC", "ppo_rb": "TR-PPO-RB",
             "ppo_espo": "ESPO", "ppo_ppg": "PPG", "ppo_exo": "ExO-PPO",
             "ppo_pbrs": "PBRS+PPO", "ppo_rnd": "RND+PPO"}
    order = list(label)
    out = []
    for task, name in (("free", "任务一 · 五指自由不打架"), ("seq", "任务二 · V→OK→OPEN 序列")):
        rows = []
        best, best_score = "--", (-1, 1e9)
        for a in order:
            e = next((e for e in summary if e["task"] == task and e["algo"] == a), None)
            if not e:
                continue
            score = (e["success_rate"], -(e.get("finish_time") or 9e9))
            if score > best_score:
                best_score, best = score, label[a]
        for a in order:
            e = next((e for e in summary if e["task"] == task and e["algo"] == a), None)
            if not e:
                continue
            star = " ⭐" if label[a] == best else ""
            def f(k, d=3):
                v = e.get(k)
                return "--" if v is None or v != v else f"{v:.{d}f}"
            rows.append(f'<tr><td>{label[a]}{star}</td><td>{e["seeds"]}</td>'
                        f'<td>{f("success_rate")}±{f("success_rate_std")}</td><td>{f("finish_time", 2)}</td>'
                        f'<td>{f("penetration_mean_mm", 2)}</td><td>{f("train_auc", 2)}</td>'
                        f'<td>{int((e.get("wall_seconds") or 0) / 60)}</td></tr>')
        out.append(f"""<h3>{name}<span class="pill">最优: {best}</span></h3>
<table><thead><tr><th>方法</th><th>种子</th><th>成功率±σ</th><th>到位/序列 (s)</th>
<th>穿透 (mm)</th><th>奖励AUC</th><th>耗时(分)</th></tr></thead><tbody>
{''.join(rows)}</tbody></table>""")
    return "\n".join(out)


def build_index():
    data = json.loads((ROOT / "hands-rl/results" / "summary.json").read_text(encoding="utf-8"))
    n = data["runs"]
    tiles = "".join(
        f'<a class="tile" href="{href}"><b>{emoji} {label}</b><span>{desc}</span></a>'
        for href, label, emoji, desc in [
            ("hardware.html", "硬件设计与安装", "🔧", "Aitachi-Hands 硬件 · ORCA v2 本体 · HLS 舵机 · 打印装配 · 17轴实测标定表"),
            ("servo.html", "舵机控制指令集", "📡", "帧格式 · 寄存器 · SYNC WRITE 0x83 · 四重安全护栏"),
            ("software.html", "软件对齐设计", "🧭", "MJCF↔舵机ID 映射 · rad/raw 换算 · 四层软件栈"),
            ("drivers.html", "驱动安装", "🔌", "COM3@1Mbps · FD.exe · C++ 原生软件 · WSL 环境"),
            ("mujoco.html", "MuJoCo 仿真教程", "🕹️", "安装 · 手势Demo · RL 环境 · 命令速查"),
            ("isaac.html", "Isaac Sim / Isaac Lab", "🚀", "Ubuntu 无界面管线 · USD 资产 · 圆棍交接 · 回放"),
            ("rl_framework.html", "RL 框架与双任务", "🎯", "任务定义 · 奖励函数 · 残差动作设计 · MDP 规格"),
            ("algos16.html", "16 算法先进性分类", "🗂️", "四组演进阶梯 · 目标函数几何 · TD3/SAC/PPO 对照"),
            ("experiments.html", "对比实验专区", "🧪", "过程·指标·流程·12图·结果表（重点）"),
            ("report.html", "实验报告（4 语言）", "📘", "English | 中文 | 日本語 | 한국어"),
            ("paper.html", "IEEE SCI 论文", "📄", "IEEEtran 双栏 · 数据自动生成"),
            ("resources.html", "资源下载", "📦", "教材 PDF · 论文 PDF · GitHub · 上游仓"),
        ])
    body = hero(
        "Aitachi-Hands",
        "开源 17 轴灵巧手全栈平台：硬件设计 · 安装标定 · 软件对齐 · 驱动 · MuJoCo / Isaac Lab 仿真 · "
        "TD3 / SAC / PPO 及 6 种 PPO 改进方法的受控对比实验") + wrap(f"""
<div class="kpi">
<div><b>17</b><span>自由度（16 指 + 腕）</span></div>
<div><b>2</b><span>RL 任务（自由不打架 / 手势序列）</span></div>
<div><b>11</b><span>算法（PPO 系 8 + TD3/SAC）</span></div>
<div><b>66</b><span>实验跑（2×11×3 · 已完成 {n}）</span></div>
<div><b>12+5</b><span>对比图 + 知识图谱矢量图</span></div>
</div>
<div class="note">本项目机械本体采用开源 <b>ORCA Hand v2</b>（CC BY 4.0，依许可署名）、飞特 HLS 串行总线舵机为执行器，
全部映射表、协议、参数、命令与实验数据均为<b>实机实测口径</b>。GitHub：
<a href="{GITHUB}">{GITHUB.replace('https://', '')}</a></div>
<h2>站点导航</h2>
<div class="grid">{tiles}</div>
<h2>两条主线</h2>
<div class="cols2">
<div class="card"><b>主线一 · 全栈教程</b><p>从图纸打印、17 舵机装配标定、COM3 串口协议，到 MuJoCo 与 Isaac Lab 双仿真线的可复制命令块。
教程与 19 页 PDF 教材同源。</p></div>
<div class="card"><b>主线二 · 受控 RL 对比</b><p>同一残差骨干、同一预算（150k 步 × 3 种子）、同一 25 评估种子下，
比较 PPO / TRPO / PPO-KL / TD3 / SAC 与 TR-PPO-RB · ESPO · PPG · ExO-PPO · PBRS · RND，
全部实现偏差如实披露。</p></div>
</div>""")
    (SITE / "index.html").write_text(page("index", "Aitachi-Hands", body, "index.html"), encoding="utf-8")


def build_others(summary):
    # ---- hardware ----
    body = hero("硬件设计与安装", "ORCA Hand v2 · 17 DOF · 双腱驱动 · 飞特 HLS 舵机实测清单") + wrap(f"""
<h2>1 · 本体与选型</h2>
<div class="card">
<p><b>ORCA Hand v2</b>（© 2026 ORCA Dexterity, CC BY 4.0）：17 DOF（16 指关节 + 1 腕）· 滚动接触关节 ·
0.4 mm 鱼线双腱 · 论文成本 &lt;2000 CHF · 单人装配 ≤8 h · 图纸 393 文件（196 STL + 85 STEP + 15 3MF）。</p>
<table><tr><th>项目</th><th>官方口径</th><th>本机执行</th></tr>
<tr><td>舵机</td><td>Feetech STS3215 / Dynamixel 二选一</td><td><b>HLS3930（腕, ID0）+ 16×HLS2915</b>，固件 3.45</td></tr>
<tr><td>总线</td><td>3 Mbps 单总线</td><td>COM3 @ 1 Mbps（实测配置）</td></tr>
<tr><td>供电</td><td>12 V</td><td>9.6–12.4 V 工作区间（护栏）</td></tr>
<tr><td>皮肤</td><td>TPU 直打 / 硅胶模具</td><td>同官方（SiliconeMolds-1100.3mf）</td></tr></table></div>
<h2>2 · 装配流程（≤8 h）</h2>
<div class="card"><ol>
<li>打印备料：PLA 结构件（或 Prints-1100.3mf 整板）+ TPU/硅胶皮肤 + 06_CNC 2 件机加工 + 轴承/M4 螺丝；</li>
<li>指节装配（01_Fingers 32 件）：滚动关节对合 → 销钉 → 穿 0.4 mm 双腱；</li>
<li>腱路：PTFE 管按 ASSEMBLY_ADDENDUM.md 裁长，绕线轮预绕，屈/伸张力一致；</li>
<li>腕/掌座（Carpals 12 · Wrist 14 · ForeArm 26）；</li>
<li>舵机入位：<b>先单舵机 USB 改 ID（1–16 指 + 0 腕）再上总线</b>；</li>
<li>电气：菊花链单总线 + 共地，12 V 供电；</li>
<li>皮肤热装/浇注。</li></ol>
<div class="warn"><b>避坑</b>：装配视频 YouTube <code>TgIz7HiyaoU</code>；PTFE 管长度直接影响腱张力，
ID13 行程极窄（±57 raw）有机械阻力史，优先检查该轴腱路。</div></div>
<h2>3 · 17 轴实测标定表（FD.exe 缓动采集）</h2>
<div class="card">
<p>标定口径：缓动至机械极限取观测区间 → 两端内缩 50 raw 作安全限位 → 中点为 center；每轴 ≥7 采样点。</p>
<table><thead><tr><th>ID</th><th>手指</th><th>机械关节</th><th>min</th><th>center</th><th>max</th><th>rel 行程</th></tr></thead><tbody>
<tr><td>0</td><td>腕</td><td>手腕前后摇摆</td><td>891</td><td>1857</td><td>2823</td><td>±966</td></tr>
<tr><td>1</td><td>无名指</td><td>一级指根（左右）</td><td>1802</td><td>2142</td><td>2483</td><td>±340</td></tr>
<tr><td>2</td><td>无名指</td><td>二级指根（前后）</td><td>784</td><td>1521</td><td>2259</td><td>±737</td></tr>
<tr><td>3</td><td>无名指</td><td>中间关节</td><td>609</td><td>1766</td><td>2924</td><td>±1157</td></tr>
<tr><td>4</td><td>中指</td><td>指尖关节</td><td>1053</td><td>1656</td><td>2260</td><td>±603</td></tr>
<tr><td>5</td><td>中指</td><td>一级指关节（左右）</td><td>1637</td><td>2016</td><td>2396</td><td>±379</td></tr>
<tr><td>6</td><td>食指</td><td>左右位移关节</td><td>1651</td><td>1926</td><td>2201</td><td>±275</td></tr>
<tr><td>7</td><td>食指</td><td>二级指根（前后）</td><td>1735</td><td>2505</td><td>3276</td><td>±770</td></tr>
<tr><td>8</td><td>食指</td><td>指尖关节（前后）</td><td>1153</td><td>2153</td><td>3153</td><td>±1000</td></tr>
<tr><td>9</td><td>小指</td><td>一级指根（左右摇摆）</td><td>1666</td><td>2057</td><td>2448</td><td>±391</td></tr>
<tr><td>10</td><td>小指</td><td>二级指根（前后）</td><td>1730</td><td>2477</td><td>3225</td><td>±747</td></tr>
<tr><td>11</td><td>小指</td><td>指尖关节</td><td>2189</td><td>2671</td><td>3153</td><td>±482</td></tr>
<tr><td>12</td><td>中指</td><td>二级指根（前后）</td><td>1624</td><td>2467</td><td>3310</td><td>±843</td></tr>
<tr><td>13</td><td>拇指</td><td>二级指根（上下）</td><td>2232</td><td>2289</td><td>2346</td><td>±57</td></tr>
<tr><td>14</td><td>拇指</td><td>三级关节（主屈轴）</td><td>1304</td><td>1708</td><td>2112</td><td>±404</td></tr>
<tr><td>15</td><td>拇指</td><td>指尖关节</td><td>1125</td><td>1993</td><td>2861</td><td>±868</td></tr>
<tr><td>16</td><td>拇指</td><td>一级指根（左右）</td><td>1899</td><td>2092</td><td>2286</td><td>±193</td></tr>
</tbody></table></div>
<h2>4 · 验收基准</h2>
<div class="card"><ul>
<li>全轴卸力回读：寄存器 40→0 后读回必须为 0（未确认 = 断电处理）；</li>
<li>三手势编码器到位容差 ±20 raw；轨迹回放允许一次 ≤40 raw 末端补正；</li>
<li>护栏：峰值电流 &lt;120 raw · 温度 &lt;40°C · 电压 9.6–12.4 V · 停滞 500 ms 切除。</li></ul></div>""")
    (SITE / "hardware.html").write_text(page("hw", "硬件", body, "hardware.html"), encoding="utf-8")

    # ---- servo ----
    body = hero("舵机控制指令集", "飞特 HLS 实测协议 · 帧格式 · 寄存器 · 三轴同步写 · 安全护栏") + wrap("""
<h2>1 · 帧格式（SCS/STS/HLS 总线族）</h2>
<div class="card"><pre>FF FF | ID | LEN | 指令 | 参数... | 校验和        校验和 = ~sum(ID..末参数) 低8位</pre>
<table><tr><th>码</th><th>指令</th><th>说明</th></tr>
<tr><td>0x01</td><td>PING</td><td>探活</td></tr>
<tr><td>0x02</td><td>READ</td><td>参数 = 起始地址 + 字节数</td></tr>
<tr><td>0x03</td><td>WRITE</td><td>参数 = 地址 + 数据</td></tr>
<tr><td>0x04/0x05</td><td>REG WRITE / ACTION</td><td>缓存写 + 触发</td></tr>
<tr><td>0x06</td><td>RESET</td><td>出厂（慎用，清 ID/零位）</td></tr>
<tr><td><b>0x83</b></td><td><b>SYNC WRITE</b></td><td><b>多轴同步写（本机 3 轴/批，广播 0xFE）</b></td></tr></table></div>
<h2>2 · 关键寄存器（实测）</h2>
<div class="card"><table><tr><th>地址</th><th>名称</th><th>要点</th></tr>
<tr><td>3 (word)</td><td>型号编码</td><td>6922=HLS2915 · 4106=HLS3930</td></tr>
<tr><td>9 / 11 (word)</td><td>最小/最大位置限位</td><td>目标越界即拒发</td></tr>
<tr><td>30 / 33</td><td>模式/分辨率</td><td>应为 1 / 0</td></tr>
<tr><td><b>40</b></td><td><b>转矩使能</b></td><td>1=上电保持 · 0=卸力（<b>卸力必须回读确认</b>）</td></tr>
<tr><td><b>41 (9B)</b></td><td><b>运动指令</b></td><td>加速度(1)+位置(2)+电流(2)+速度(2)+扭矩(2)，小端</td></tr>
<tr><td><b>56 (15B)</b></td><td><b>反馈区</b></td><td>[0:2]位置 [2:4]速度 [6]电压 [7]温度 [13:15]电流</td></tr></table>
<p>位置/速度/电流为<b>符号-数值 15 位编码</b>（最高位符号）。</p></div>
<h2>3 · 三轴同步写示例（实测配方）</h2>
<div class="card"><pre>FF FF FE 22 83 29 09            # 广播FE · LEN=4+10*3 · 起始41 · 每轴9字节
  &lt;ID1&gt; &lt;accel&gt; &lt;pos_L pos_H&gt; &lt;cur_L cur_H&gt; &lt;spd_L spd_H&gt; &lt;lim_L lim_H&gt;
  &lt;ID2&gt; ...                      # 每批最多3轴 · ID 不得重复
  &lt;ID3&gt; ...
  &lt;checksum = ~sum(ID..lim_H)&gt;
<b>合法性窗口</b>：位置 50–4050 · 加速度 1–254 · 速度 1–600 · 电流 1–68 · 扭矩 1–1000
<b>闭环</b>：发送后逐轴回读41比对 → 轮询56直到 |位置-目标|≤20 且 |速度|≤2 连续3次</pre></div>
<h2>4 · 四重安全护栏（C++ guard 实测阈值）</h2>
<div class="card"><table><tr><th>护栏</th><th>阈值</th><th>动作</th></tr>
<tr><td>峰值电流</td><td>≥120 raw</td><td>立即切除</td></tr>
<tr><td>温度</td><td>≥40 °C</td><td>立即切除</td></tr>
<tr><td>电压</td><td>出 9.6–12.4 V</td><td>立即切除</td></tr>
<tr><td>静止高电流</td><td>≥77 且 |速度|≤2 持续500ms</td><td>切除（防堵转）</td></tr>
<tr><td>无进展</td><td>误差&gt;20 超1s无收窄</td><td>切除</td></tr></table></div>
<h2>5 · 上位机与低层调用</h2>
<div class="card"><pre><b>FD.exe</b>（飞特官方，提取码 juds）：改ID / 调零位PID / 缓动标定 / 示波回读（与自研软件互斥 COM3）
<b>AHandNative.exe --inspect</b>     # 巡检：连17轴、识别当前姿态、不发运动
<b>AHandNative.exe --run V</b>       # 执行 OPEN→V（结束自动卸力回读）
<b>AHandNative.exe --selftest</b>    # 协议/护栏自检（不开串口）
<b>python align_real.py --status</b> # 只读巡检 · --dry-run 干跑 · --probe 14 +60 单轴探针</pre>
<div class="warn">手册入口 doc.feetech.cn（FTServo e-manual）；FD 的零位偏移写 EEPROM，本项目标定只读不写。</div></div>""")
    (SITE / "servo.html").write_text(page("servo", "舵机指令", body, "servo.html"), encoding="utf-8")

    # ---- software ----
    body = hero("软件对齐设计", "同一只手的四层表达：FD 上位机 → 网页控制 → 原生 C++ → 仿真/RL") + wrap("""
<h2>1 · 四层软件栈</h2>
<div class="card"><table><tr><th>层</th><th>工具</th><th>角度表示</th><th>职责</th></tr>
<tr><td>上位机</td><td>FD.exe</td><td>raw 0–4095</td><td>改ID/零位/PID · 缓动标定</td></tr>
<tr><td>网页控制</td><td>hand-control.py + HTML</td><td>raw 目标+档位</td><td>三轴并行批次 · 电流/温度暂停</td></tr>
<tr><td>原生 GUI</td><td>AHandNative.cpp (Win32)</td><td>raw+编译期轨迹库</td><td>SYNC WRITE · 四重保护 · 调优</td></tr>
<tr><td>仿真/RL</td><td>MuJoCo / Isaac Lab</td><td>rad 关节角</td><td>训练评估 · 导出真机目标</td></tr></table></div>
<h2>2 · 关节 ↔ 舵机映射（2026-11-11 位姿差分实测）</h2>
<div class="card"><table><thead><tr><th>MJCF 关节</th><th>舵机 ID</th><th>实测关节名</th><th>rel 行程</th></tr></thead><tbody>
<tr><td>right_wrist</td><td>0</td><td>腕前后摇摆</td><td>±966</td></tr>
<tr><td>right_t-abd / t-cmc / t-mcp / t-pip</td><td>16 / 13 / 14 / 15</td><td>拇指（14=主屈轴）</td><td>±193 / ±57 / ±404 / ±868</td></tr>
<tr><td>right_i-abd / i-mcp / i-pip</td><td>6 / 7 / 8</td><td>食指</td><td>±275 / ±770 / ±1000</td></tr>
<tr><td>right_m-abd / m-mcp / m-pip</td><td>5 / 12 / 4</td><td>中指</td><td>±379 / ±843 / ±603</td></tr>
<tr><td>right_r-abd / r-mcp / r-pip</td><td>1 / 2 / 3</td><td>无名指</td><td>±340 / ±737 / ±1157</td></tr>
<tr><td>right_p-abd / p-mcp / p-pip</td><td>9 / 10 / 11</td><td>小指</td><td>±391 / ±747 / ±482</td></tr>
</tbody></table></div>
<h2>3 · rad ↔ raw 换算</h2>
<div class="card"><pre>q_i  = s_i · (raw_i − center_i)/(max_i − min_i) · Δq_i^max      # Δq 取 MJCF jnt_range
raw_i = center_i + s_i · q_i/Δq_i^max · (max_i − min_i)         # s_i 方向由 A/B 位姿差分定</pre>
<p>锚点位姿（V/OK/OPEN 的 raw 目标表）做最小二乘定标 → 摄像头指尖跟踪矫正残差。</p></div>
<h2>4 · 系统数据流</h2>
<div class="card"><pre>FD 标定 ─→ AHand-calibration.json ─→ 锚点位姿表 ─→ C++ 轨迹库 / 网页控制 ─→ <b>实机 COM3@1M</b>
                     ↓                                                        ↑ 电流/温度/位置回读护栏
              MuJoCo / Isaac Lab (rad) ─→ RL 双任务 ─→ 导出轨迹+分段回放 ────┘</pre></div>""")
    (SITE / "software.html").write_text(page("sw", "软件对齐", body, "software.html"), encoding="utf-8")

    # ---- drivers ----
    body = hero("驱动安装", "Windows 实机线 + WSL 训练线，双线独立") + wrap("""
<h2>1 · Windows 实机控制线</h2>
<div class="card"><ol>
<li>USB 转串口驱动（CP210x/CH340 按转接板芯片）→ 设备管理器确认 <b>COM3</b>，波特率 <b>1 000 000</b>；</li>
<li>FD.exe 上位机（网盘提取码 juds）——标定/改号；<b>与自研软件互斥 COM3</b>；</li>
<li>AHandNative.exe 免安装：按钮 = 连接检查/V/OK/OPEN/停止卸力 + 三姿势四档位调优（50–100%）；
命令行 <code>--selftest / --inspect / --run V|OK|OPEN</code>；源码 <code>build.cmd</code>（MSVC）；</li>
<li>Python 网页控制：hand-control.py（pyserial）+ 启动*.cmd；快速手势面板 fast-gesture-panel.html。</li></ol>
<div class="warn"><b>安全铁律</b>：异常先"停止并卸力"（寄存器40→0 回读确认）；卸力未确认 = 立即断舵机电源。</div></div>
<h2>2 · WSL 训练线（Ubuntu-22.04）</h2>
<div class="card"><pre>cd /mnt/c/Users/ASUS/Desktop/AHand/style1101
bash scripts/setup_ubuntu.sh            # 离线 wheels 优先
.venv/bin/python -m unittest discover -s tests -v
# 栈：torch 2.11.0+cu130 · SB3 2.9.2 · MuJoCo 3.3.7 · gymnasium 1.4.0</pre>
<p>共享文件 ≠ 共享 Python 环境：Windows 实机线与 WSL 训练线依赖各自独立；Isaac 用
<code>$HOME/isaac_sim/venv/bin/python</code>。</p></div>""")
    (SITE / "drivers.html").write_text(page("drv", "驱动", body, "drivers.html"), encoding="utf-8")

    # ---- mujoco ----
    body = hero("MuJoCo 仿真教程", "官方 orca_v2 场景 · 手势浏览器 Demo · RL 环境入口 · 命令速查") + wrap("""
<h2>1 · 安装与模型</h2>
<div class="card"><pre>pip install mujoco==3.3.7
python -m mujoco.viewer --mjcf=orca_v2/scene_right.xml     # 官方描述仓 v2 目录
# GPU 渲染需 EGL/OSMesa；纯数值训练与评估不依赖 OpenGL（渲染失败≠训练失败）</pre></div>
<h2>2 · 手势浏览器 Demo（实测配方）</h2>
<div class="card"><pre>cd AHand/mujoco
./.venv/Scripts/python.exe serve_demo.py --open    # 浏览器 http://127.0.0.1:8767
<b>按键</b>：1/V=V · 2/O=OK · 0=张开 · A=自动 · 空格=暂停 · R=正视
gesture_demo.py --test                              # 纯物理自检（无浏览器）
serve_demo.py --port 8768                           # 端口被占时</pre></div>
<h2>3 · RL 环境（本仓 envs/）</h2>
<div class="card"><pre>import gymnasium as gym
import envs                                        # 注册 AitachiFree-v1 / AitachiSeq-v1
env = gym.make("AitachiSeq-v1"); obs, info = env.reset(seed=7001)
# 执行器 kp=12 kv=0.5（腕 60/3）· 物理 0.002s · 控制 0.02s · 隐式快速积分</pre></div>
<h2>4 · 命令速查</h2>
<div class="card"><table><tr><th>目的</th><th>命令</th></tr>
<tr><td>原生查看器</td><td><code>python -m mujoco.viewer --mjcf=scene_right.xml</code></td></tr>
<tr><td>浏览器 Demo</td><td><code>serve_demo.py --open</code></td></tr>
<tr><td>无头自检</td><td><code>gesture_demo.py --test</code></td></tr>
<tr><td>RL 冒烟</td><td><code>run_matrix.py --task free --algo ppo --timesteps 2048</code></td></tr>
<tr><td>回放 GIF</td><td><code>MUJOCO_GL=egl demo.py V --model best_model.zip</code></td></tr></table>
<div class="warn">WSL Vulkan 仅 llvmpipe：原生窗口不可行 → EGL 离屏或浏览器服务渲染；回放失败须如实记录。</div></div>""")
    (SITE / "mujoco.html").write_text(page("mj", "MuJoCo", body, "mujoco.html"), encoding="utf-8")

    # ---- isaac ----
    body = hero("Isaac Sim / Isaac Lab 教程", "Ubuntu·WSL 无界面管线 · CUDA PhysX · 圆棍交接 · FFplay 回放") + wrap("""
<h2>1 · 数据流与路径</h2>
<div class="card"><pre>URDF/STL ─→ 物理 USD（固定底座+17关节） ─→ Isaac Lab（CUDA PhysX+测量JSON） ─→ FFplay/USDA 回放
# 项目目录 /mnt/c/Users/ASUS/Desktop/AHand · 模型 1028/orcahand_description/v2
# Isaac Python $HOME/isaac_sim/venv/bin/python · Kit ubuntu-validation/isaac_physics.kit
# 硬件 RTX 3070 Laptop（CUDA 可用）· WSL Vulkan=llvmpipe ⇒ 无界面求解+本地工具回放</pre></div>
<h2>2 · 依赖自检（整块复制）</h2>
<div class="card"><pre>(
export AHAND_ROOT=/mnt/c/Users/ASUS/Desktop/AHand/gpu-validation
export AHAND_PY="$HOME/isaac_sim/venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES
cd "$AHAND_ROOT" || exit 1
mkdir -p logs stick-grasp/runs || exit 1
/usr/lib/wsl/lib/nvidia-smi -L
test -x "$AHAND_PY" && echo "Isaac Python exists"
test -f usd/right/orca_right.usd && test -f usd/left/orca_left.usd && echo "Both USD assets exist"
)</pre></div>
<h2>3 · 主实验与回放</h2>
<div class="card"><pre>bash ./run_isaac_orca_wsl.sh                 # 左右手 GPU 动力学（600步 · 回位误差0.01783rad）
bash ./run_isaac_cylinder_handoff_wsl.sh     # 圆柱交接
bash ./run_isaac_stick_grasp_wsl.sh          # 圆棍抓取
tail -n 35 logs/handoff-latest.log           # 运行中另开终端看日志

OPENBLAS_NUM_THREADS=1 "$AHAND_PY" stick-grasp/render_ubuntu_handoff.py \\
  --track stick-grasp/pad-sway-cushion.json --output Ubuntu-ORCA-Handoff.mp4 &&
ffplay -autoexit Ubuntu-ORCA-Handoff.mp4      # 空格暂停 · Q 退出</pre>
<p>已验证：抬升约 17.00 mm · 摇摆最大漂移约 7.97 mm（超验收记失败态）· 455 帧轨迹 JSON。</p></div>
<h2>4 · 实机 A/B 与仿真对照</h2>
<div class="card"><p>确认版教程 #hardware 节内嵌实机 A/B 历史串口采样：选轴离线绘制 raw 位置曲线（虚线=目标）、
电流/负载/温度，并有目标越界复核表——仿真角度对齐实机编码的日常工具。</p>
<div class="note">MuJoCo 的 V/OK 用模型关节目标；Isaac Lab 主线为圆棍交接。两线共享几何，物理引擎不同，跨引擎结论分别验证。</div></div>""")
    (SITE / "isaac.html").write_text(page("is", "Isaac", body, "isaac.html"), encoding="utf-8")

    # ---- rl framework ----
    body = hero("RL 框架与双任务", "MuJoCo 双任务 MDP · 奖励函数 · 残差动作设计") + wrap(f"""
<h2>1 · 任务一 AitachiFree-v1：五指全自由活动但不打架</h2>
<div class="card">
<p>每回合在三锚点（OPEN/V/OK）的 <b>Dirichlet 凸组合 + σ=0.05 rad 噪声</b>流形上采样目标。
成功 = 位置误差 ≤0.12 rad + 速度 ≤0.35 rad/s + <b>穿透 ≤1 mm 保持 3 个控制周期</b>；
穿透 &gt;5 mm 为失败线。"不打架"是显式成功判据。预算 2.5 s，控制 16 指（腕固定），观测 80 维。</p></div>
<h2>2 · 任务二 AitachiSeq-v1：V→OK→OPEN 快速序列</h2>
<div class="card">
<p>三相位接力，每相位需 v3 验收窗（位置+速度+几何：V 分指 ≥30mm · OK 拇食距 ≤10mm）保持 3 周期才切换。
相位奖 8+8×剩余比例，终段 +5。预算 3×2.0 s；观测 84 = 80 + 相位 one-hot 3 + 进度 1。</p></div>
<h2>3 · 奖励函数（v3 数值验证权重）</h2>
<div class="card"><pre>r = −4·mean(e_norm²) − 0.02·mean(v_excess²) − 0.005·mean(Δa²) − 0.01·mean(f_norm²)
    − 2·coll − 0.02·time + 2·(ē_prev − ē_t) + 0.8·1_tol + r_phase
    + (10 + 10·(1 − t/T))·1_succ − 2·1_fail
<b>碰撞不提前终止</b>（持续惩罚）——早退自杀曾被策略利用逃避时间惩罚</pre></div>
<h2>4 · 残差动作设计（公平性核心）</h2>
<div class="card"><pre>指令 = 归一化(目标先验 + 0.3×π(s))，每周期限速 0.15 rad
<b>零策略 = 可解直接基线</b> → 所有算法在同一正确先验上优化"更快+更稳+低压+低穿透"</pre>
<div class="note">历史教训（v2→v3 三重根因）：增量动作随机游走 / 采样噪声破坏保持判据 / 碰撞早退被利用，
修复后 150k 步三手势成功率 1.00、到位 0.39–0.46 s。</div></div>
<h2>5 · 代码入口</h2>
<div class="card"><pre>envs/hand_core.py      # 模型加载/执行器/关节索引/metrics（v3 同口径）
envs/free_task.py      # 任务一（自然位姿流形采样 + 无碰撞保持判据）
envs/seq_task.py       # 任务二（三相位自动机 + 相位分解计时）
algorithms/baselines.py    # PPO/TD3/SAC 超参表
algorithms/ppo_x.py        # PPOX：clip/rb/exo 目标 + ESPO 早停 + PPG 相位
algorithms/reward_addons.py# PBRS 势能塑形 + RND 内在奖励</pre></div>""")
    (SITE / "rl_framework.html").write_text(page("rl", "RL框架", body, "rl_framework.html"), encoding="utf-8")

    # ---- algos16 ----
    fig_obj = "figs/fig06_ppo_diagnostics.png"
    body = hero("16 篇论文先进性分类", "四组演进阶梯 · 11 算法实跑（含 TRPO/PPO-KL）· ACER 离散对照 · 目标函数几何") + wrap(f"""
<h2>1 · 四组先进性阶梯</h2>
<div class="card"><table><tr><th>组</th><th>论文</th><th>机制一句话</th><th>本项目角色</th></tr>
<tr><td><b>A 基础与复现</b></td><td>01 TRPO · 02 GAE · 03 PPO · 04 Impl-Matters · 05 OnPolicy-AC</td>
<td>信赖域→裁剪代理；实现细节贡献 17–21%</td><td>03 实跑基线；04/05 定义公平协议</td></tr>
<tr><td><b>B PPO 算法改进</b></td><td>01 TRPO(CG+线搜) · PPO-KL(自适应惩罚) · 06 TR-PPO-RB · 07 ESPO · 08 PPG · 25 ExO-PPO · 26 OPPO</td>
<td>越界反向斜率/漂移早停/相位分离/平滑压缩/服务重叠</td><td><b>TRPO/PPO-KL/06/07/08/25 实跑</b>；26 引用（无 LLM 服务环）</td></tr>
<tr><td><b>C 环境奖励与探索</b></td><td>09 PBRS · 10 Human-Pref · 11 RND · 12 Eureka · 13 TWTL · 27 Improve-RM</td>
<td>势能不变性/偏好奖励/预测误差内在奖励/LLM 奖励搜索/时序逻辑/主动偏好</td><td><b>09/11 实跑</b>；其余需人标或 LLM，引用分类</td></tr>
<tr><td><b>D 落地参照</b></td><td>57 ACER(离散对照) · 31 TD3 · 32 SAC · 56 ANYmal · 55 RMA · 62 RL-100</td>
<td>双Q防过估/最大熵/执行器建模/快速适应/蒸馏部署</td><td>31/32 实跑基线；ACER 因 SB3 仅离散无法连续实跑（见跨研究表）；其余参照</td></tr></table></div>
<h2>2 · 目标函数几何（clip vs rollback vs ExO）</h2>
<div class="card">
<img class="fig" src="{fig_obj}" alt="diagnostics">
<p class="figcap">上图：三种变体的诊断统计对比（clip 率 / 近似 KL / 最大比率 / 奖励 AUC）。
教材第 6 章含 ρ-目标几何矢量图：PPO-Clip 越界梯度=0；TR-PPO-RB 反向斜率拉回；ExO 平滑压缩不归零。</p></div>
<h2>3 · 三基线一页纸</h2>
<div class="card"><table><tr><th></th><th>TD3 (2018)</th><th>SAC (2018)</th><th>PPO (2017)</th></tr>
<tr><td>类别</td><td>离策略确定性 AC</td><td>离策略最大熵随机 AC</td><td>在策略随机 PG</td></tr>
<tr><td>核心</td><td>双Q取小+目标平滑+延迟actor</td><td>熵正则+双Q+自动温度</td><td>裁剪比率+GAE+多轮小批</td></tr>
<tr><td>探索</td><td>外加截断噪声</td><td>熵驱动内生</td><td>高斯采样</td></tr>
<tr><td>本工程超参</td><td>lr 1e-3 · batch 256 · 噪声0.1 · 延迟2</td><td>lr 3e-4 · batch 256 · 目标熵−16</td><td>lr 3e-4 · 8×256/批 · 10轮</td></tr></table></div>
<h2>4 · 实现偏差披露</h2>
<div class="card"><ol>
<li>TR-PPO-RB：逐状态 KL 用 |log ρ|&gt;0.5 代理（SB3 不存旧分布参数），c=(1+ε)/(1−ε)=1.5；</li>
<li>RND：单价值头简化（原文双头+非回合内在回报），λ=0.5 滑动归一；</li>
<li>ExO-PPO：ε·tanh 软饱和复现"平滑压缩"；</li>
<li>OPPO：仅引用（LLM 服务系统论文）；PBRS 超时截断不加终端势修正（成功时 Φ≈0）。</li></ol></div>""")
    (SITE / "algos16.html").write_text(page("a16", "16算法", body, "algos16.html"), encoding="utf-8")

    # ---- experiments (KEY page) ----
    figs = [
        ("fig01_learning_free.png", "任务一训练曲线（平滑回合奖励 × 环境步；3 种子均值）"),
        ("fig02_learning_seq.png", "任务二训练曲线"),
        ("fig03_final_success.png", "最终成功率（确定性 25 集评估，均值±标准差）"),
        ("fig04_finish_time.png", "到位/序列时间对比"),
        ("fig05_sample_efficiency.png", "样本效率：达到最终奖励 90% 所需步数"),
        ("fig06_ppo_diagnostics.png", "PPO 族诊断：clip 率 / 近似 KL / 最大比率 / 奖励 AUC"),
        ("fig07_espo_stopping.png", "ESPO 早停：KL 轨迹（×=早停点）与停止轮分布"),
        ("fig08_reward_group.png", "奖励塑形组（PBRS/RND vs PPO）与 RND 内在量尾部"),
        ("fig09_eval_box.png", "评估集到位/序列时间分布（箱线）"),
        ("fig10_phase_decomposition.png", "任务二相位时间分解（V/OK/OPEN）"),
        ("fig11_wallclock.png", "墙钟效率：最终成功率 vs 训练时长"),
        ("fig12_safety_penetration.png", "安全代理指标：评估期最大自穿透"),
    ]
    fig_html = "".join(
        f'<img class="fig" src="figs/{f}" alt="{c}"><p class="figcap">{c}</p>'
        for f, c in figs)
    body = hero("对比实验专区", "过程 · 指标 · 流程 · 12 张对比图 · 实时结果表") + wrap(f"""
<h2>1 · 实验矩阵与流程</h2>
<div class="card">
<div class="kpi"><div><b>2</b><span>任务</span></div><div><b>9</b><span>算法</span></div>
<div><b>3</b><span>种子</span></div><div><b>66</b><span>总跑数</span></div>
<div><b>150k</b><span>步/跑</span></div></div>
<pre>流程：<b>同残差骨干环境</b> → 8 并行采样（PPO 族）/ 单环境回放（TD3/SAC）
   → 训练 150k 步（Monitor 逐集日志 + PPO 诊断）
   → <b>固定 25 评估种子（7001–7025）确定性复测</b> → aggregate → 12 图 → 论文/README 自动生成</pre>
<p>每次运行目录含 config.json（模型/目标 SHA-256、版本、GPU）、episodes.csv、
ppo_diagnostics.json（比率/clip/KL/早停/辅助损失）、eval.json + eval_episodes.csv。</p></div>
<h2>2 · 指标定义</h2>
<div class="card"><table><tr><th>指标</th><th>定义</th></tr>
<tr><td>成功率</td><td>25 个固定种子确定性评估中满足全部判据的回合占比</td></tr>
<tr><td>到位/序列时间</td><td>任务一首次进入保持窗的时刻；任务二三相位完成总时（回溯到保持窗首周期）</td></tr>
<tr><td>穿透 (mm)</td><td>回合内最大自碰撞穿透（安全代理指标）</td></tr>
<tr><td>奖励 AUC</td><td>平滑回合奖励曲线归一化面积（越接近 1 = 越早达到终水平）</td></tr>
<tr><td>诊断量</td><td>clip 率 / 近似 KL / 最大比率 / ESPO 停止轮占比 / PPG 辅助 KL</td></tr></table>
<div class="warn"><b>训练曲线为何用奖励不用成功率</b>：残差骨干下采样噪声结构性打破保持窗，
训练期成功率≈0 是统计假象（确定性评估满分）——已在报告如实披露。</div></div>
<h2>3 · 结果表（随 results/summary.json 实时更新）</h2>
<div class="card">
{results_tables(summary)}
<p class="figcap">生成命令：<code>python scripts/aggregate.py &amp;&amp; python scripts/gen_site.py</code></p></div>
<h2>4 · 12 张对比学术图</h2>
<div class="card">{fig_html}</div>
<h2>5 · 跨研究对照（RLP 基准）</h2>
<div class="card">
| 方法 | E1 | E2 | E3 | 注记 |
|---|---|---|---|---|
| PPG | 0.669±0.02 | 0.681 | 1.000±0.00 | |
| PPO-KL | 0.658 | 0.785 | 0.992 | |
| PPO-RB | 0.655 | 0.669 | 0.995 | |
| PPO-Clip | 0.652 | 0.620 | 0.996 | E2 种子散布 ±0.17 |
| ESPO | 0.648 | 0.755 | 0.568 | |
| TR-PPO-RB | 0.625 | 0.754 | 0.694 | |
| TRPO | -- | -- | 1.000 | E3=1.000 真值；CG+线搜开销 |
| ACER | -- | -- | -- | 离散对照（SB3 ACER 仅离散动作） |
<p class="figcap">来源：Reinforcement-Learning-Policy 仓实验报告（11 页，d229af8）。ACER 在 SB3 中仅支持离散动作空间，无法在本工程连续 16 维手势任务上实跑，作为离散对照与分类引用保留。</p>
<img class="fig" src="figs/fig13_cross_study.png" alt="cross-study"><p class="figcap">fig13 · 跨研究对比图</p></div>
<h2>6 · 复现命令</h2>
<div class="card"><pre>git clone {GITHUB}.git && cd Aitachi-Hands
python scripts/fetch_assets.py           # 官方模型（111MB，不入仓）
bash scripts/smoke_all.sh                # 13 路径冒烟 ~4min
bash scripts/run_matrix_all.sh           # 66 跑全矩阵 ~6-8h（RTX 3070；trpo/ppo_kl 已并入）
python scripts/regen_all.sh              # summary + README + 论文数字 + 12 图 一键再生成</pre></div>""")
    (SITE / "experiments.html").write_text(page("exp", "实验", body, "experiments.html"), encoding="utf-8")

    # ---- report / paper / resources ----
    body = hero("实验报告（4 语言）", "English | 中文 | 日本語 | 한국어 —— GitHub README 即正式实验报告") + wrap(f"""
<div class="card"><p>仓库 <a href="{GITHUB}#readme">README.md</a> 即四语言正式实验报告（由
<code>scripts/gen_readme.py</code> 从 summary.json 自动渲染，含协议、偏差披露、局限与一键复现）。</p>
<p><a class="pill" href="{GITHUB}#english">English</a>
<a class="pill" href="{GITHUB}#中文">中文</a>
<a class="pill" href="{GITHUB}#日本語">日本語</a>
<a class="pill" href="{GITHUB}#한국어">한국어</a></p></div>
<h2>报告口径要点</h2>
<div class="card"><ul>
<li>矩阵：2 任务 × 11 方法 × 3 种子 = 66 跑 × 150k 步（含 TRPO CG+线搜、PPO-KL 自适应惩罚）；</li>
<li>共享骨干：残差动作（零策略=可解直接基线）；网络统一 256×256（PPG 共享主干）；</li>
<li>评估：确定性策略 + 25 固定种子；成功率/时间/穿透为硬指标；</li>
<li>栈：MuJoCo 3.3.7 · torch 2.11.0+cu130 · SB3 2.9.2 · RTX 3070 Laptop；</li>
<li>偏差：KL 代理 / RND 单头 / ExO tanh 复现 —— 全部披露；</li>
<li>局限：仿真口径；真机需受护栏分段回放管线；单预算受控对比非穷举。</li></ul></div>""")
    (SITE / "report.html").write_text(page("rep", "报告", body, "report.html"), encoding="utf-8")

    body = hero("IEEE SCI 论文", "IEEEtran 双栏 · 结果由 summary.json 自动生成") + wrap("""
<div class="card">
<p><b>Aitachi-Hands: An Open 17-DOF Dexterous-Hand Platform and a Controlled Study of
Proximal Policy Optimization Improvements for Gesture Control</b></p>
<embed src="Aitachi-Hands-IEEE.pdf" type="application/pdf" width="100%" height="820px">
<p class="figcap">在线预览（下载：<a href="Aitachi-Hands-IEEE.pdf">Aitachi-Hands-IEEE.pdf</a>）·
LaTeX 源：<code>paper/ieee_main.tex</code> + <code>paper/results_auto.tex</code>（自动生成）。
刷新数字：<code>python scripts/gen_results_snippet.py</code> 后重编译。</p></div>""")
    (SITE / "paper.html").write_text(page("pap", "论文", body, "paper.html"), encoding="utf-8")

    body = hero("资源下载", "教材 · 论文 · 图表 · 代码 · 上游") + wrap(f"""
<div class="card"><table><tr><th>资源</th><th>链接</th></tr>
<tr><td>🎓 全栈教材 PDF（19 页，中文）</td><td><a href="Aitachi-Hands-Textbook.pdf">本地下载</a> · 源 <code>textbook/main.tex</code></td></tr>
<tr><td>📄 IEEE 论文 PDF</td><td><a href="Aitachi-Hands-IEEE.pdf">本地下载</a> · 源 <code>paper/ieee_main.tex</code></td></tr>
<tr><td>💻 代码仓</td><td><a href="{GITHUB}">{GITHUB.replace('https://', '')}</a></td></tr>
<tr><td>🧩 官方图纸仓</td><td><a href="https://github.com/orcahand/orcahand_hardware">orcahand_hardware</a>（578MB CAD/STL/STEP/3MF）</td></tr>
<tr><td>🧩 官方描述仓</td><td><a href="https://github.com/orcahand/orcahand_description">orcahand_description</a>（URDF/MJCF/网格）</td></tr>
<tr><td>📐 ORCA 论文</td><td><a href="https://arxiv.org/abs/2504.04259">arXiv:2504.04259</a></td></tr>
<tr><td>📖 舵机手册</td><td><a href="http://doc.feetech.cn">doc.feetech.cn</a>（FTServo e-manual）· FD.exe 提取码 juds</td></tr>
<tr><td>🚀 装配视频</td><td>YouTube <code>TgIz7HiyaoU</code></td></tr></table></div>
<h2>许可证</h2>
<div class="card"><p>本仓代码/文档 MIT；机械设计参照 ORCA Hand（CC BY 4.0）；舵机通信遵循厂商公开协议文档。</p></div>""")
    (SITE / "resources.html").write_text(page("res", "资源", body, "resources.html"), encoding="utf-8")


def main():
    SITE.mkdir(exist_ok=True)
    figs = SITE / "figs"
    figs.mkdir(exist_ok=True)
    for png in (ROOT / "figures").glob("*.png"):
        shutil.copy2(png, figs / png.name)
    for src, dst in ((ROOT / "textbook" / "main.pdf", SITE / "Aitachi-Hands-Textbook.pdf"),
                     (ROOT / "paper" / "ieee_main.pdf", SITE / "Aitachi-Hands-IEEE.pdf")):
        if src.is_file():
            shutil.copy2(src, dst)
    summary = json.loads((ROOT / "hands-rl/results" / "summary.json").read_text(encoding="utf-8"))["summary"]
    build_index()
    build_others(summary)
    print("site built:", SITE, "pages:", len(list(SITE.glob("*.html"))))


if __name__ == "__main__":
    main()
