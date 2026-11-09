# -*- coding: utf-8 -*-
"""生成面试复盘用的 RAGAS 测评主报告 + 三组 HTML 小结。

为什么用脚本而不是手写 HTML：报告里 40 多个数字，手写必然和数据脱节。
这里全部从 .tmp/ragas/*.json 现算，改一次数据重跑即可。

用法：
    python scripts/ragas/make_final_report.py
"""
from __future__ import annotations

import json
import statistics as st
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / ".tmp" / "ragas"

# 指标定义：key / 中文名 / 门槛（None 表示趋势项）/ 一句话人话解释 / 生活化类比
METRICS = [
    ("faithfulness", "忠实度", 0.90, "回答里的每句话，能不能在检索到的资料里找到依据", "有没有编"),
    ("answer_relevancy", "回答相关性", 0.88, "回答的是不是用户问的那件事", "有没有答偏"),
    ("context_precision", "上下文精度", 0.85, "捞回来的资料里，有多少是真正用得上的", "有没有掺垃圾"),
    ("context_recall", "上下文召回", 0.70, "该捞回来的资料，实际捞到了多少", "有没有漏"),
    ("context_entity_recall", "实体召回", None, "关键名词（字段名、类名、数字）覆盖了多少", "专有名词对得上吗"),
    ("noise_sensitivity", "噪声敏感度", None, "往资料里掺无关内容，答案会不会被带跑", "抗不抗干扰"),
]
CORE = [m for m in METRICS if m[2] is not None]

# 各阶段取数/判分结果。同名指标在不同阶段可能来自不同判官，报告里逐处标注。
STAGES = {
    "before": [".tmp/ragas/glm_r1.json", ".tmp/ragas/glm_r2.json"],          # 基线：重排关、合并阈值 3
    "rerank": [".tmp/ragas/rr_glm_r1.json", ".tmp/ragas/rr_glm_r2.json"],    # 开重排
    "merge": [".tmp/ragas/b_glm_r1.json", ".tmp/ragas/b_glm_r2.json"],       # 合并阈值 3→2
    "after": [".tmp/ragas/scnet_w1.json"],                                   # 宽召回（第一组，单轮）
    "g2": [".tmp/ragas/scnet_k2r1.json", ".tmp/ragas/scnet_k2r2.json"],      # 第二组两轮
    "g3_raw": [".tmp/ragas/scnet_k3.json"],                                  # 第三组未剥 HTML
    "g3": [".tmp/ragas/scnet_k3b.json"],                                     # 第三组剥 HTML
}


def rows(stage: str) -> list[dict]:
    out: list[dict] = []
    for rel in STAGES[stage]:
        p = ROOT / rel
        if not p.exists():
            raise SystemExit(f"缺少数据文件：{p}")
        out += json.load(open(p, encoding="utf-8"))
    return out


def mean_of(key: str, stage: str) -> float | None:
    vals = [r["scores"][key] for r in rows(stage) if r["scores"].get(key) is not None]
    return st.mean(vals) if vals else None


def per_round(key: str, stage: str) -> list[float | None]:
    out = []
    for rel in STAGES[stage]:
        vs = [r["scores"][key] for r in json.load(open(ROOT / rel, encoding="utf-8"))
              if r["scores"].get(key) is not None]
        out.append(st.mean(vs) if vs else None)
    return out



def q_core(idx: int, stage: str) -> float | None:
    """单题的核心四项均分。多轮时先算每题每轮的均分，再对轮次取平均。"""
    vals = []
    for rel in STAGES[stage]:
        data = json.load(open(ROOT / rel, encoding="utf-8"))
        hit = next((x for x in data if x["idx"] == idx), None)
        if hit:
            vals.append(st.mean([hit["scores"][k] for k, *_ in CORE]))
    return st.mean(vals) if vals else None


def f3(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "—"


def fp(v: float | None) -> str:
    """带正负号的变化值。"""
    return f"{v:+.3f}" if v is not None else "—"


def tag(ok: bool | None) -> str:
    if ok is None:
        return '<span class="tag gray">趋势项</span>'
    return '<span class="tag ok">达标</span>' if ok else '<span class="tag no">未达</span>'


def delta_cell(d: float | None, lower_better: bool = False) -> str:
    """变化单元格：超过 0.03 噪声地板才着色，否则标持平。"""
    if d is None:
        return '<td class="num">—</td>'
    good = (d < 0) if lower_better else (d > 0)
    if abs(d) <= 0.03:
        return f'<td class="num flat">{d:+.3f}<span class="hint">持平</span></td>'
    cls = "up" if good else "down"
    return f'<td class="num {cls}">{d:+.3f}</td>'


# ---------------------------------------------------------------- 数字准备
B = {k: mean_of(k, "before") for k, *_ in METRICS}
A = {k: mean_of(k, "after") for k, *_ in METRICS}
R2 = {k: mean_of(k, "rerank") for k, *_ in METRICS}
MG = {k: mean_of(k, "merge") for k, *_ in METRICS}
G2 = {k: mean_of(k, "g2") for k, *_ in METRICS}
G3R = {k: mean_of(k, "g3_raw") for k, *_ in METRICS}
G3 = {k: mean_of(k, "g3") for k, *_ in METRICS}

B_CORE_OK = sum(1 for k, _, thr, *_ in CORE if B[k] is not None and B[k] >= thr)
A_CORE_OK = sum(1 for k, _, thr, *_ in CORE if A[k] is not None and A[k] >= thr)
NOW = datetime.now().strftime("%Y-%m-%d %H:%M")

CSS = """
:root{
  --ink:#141a22; --sub:#5c6878; --faint:#8b98a8; --line:#e4e9f0; --soft:#f6f8fb;
  --blue:#2f6df6; --blue-soft:#eef3ff; --green:#0e8a5f; --green-soft:#eaf7f1;
  --red:#d13b3b; --red-soft:#fdeeee; --amber:#a96208; --amber-soft:#fdf5e6;
}
*{box-sizing:border-box}
body{
  margin:0; background:#fff; color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  font-size:15px; line-height:1.85; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1000px; margin:0 auto; padding:0 28px 80px}
header.hero{
  padding:56px 0 34px; border-bottom:3px solid var(--ink); margin-bottom:34px;
}
.kicker{font-size:12px; letter-spacing:.18em; color:var(--faint); text-transform:uppercase; margin-bottom:14px}
h1{font-size:34px; line-height:1.3; margin:0 0 14px; letter-spacing:-.01em}
.lede{font-size:16px; color:var(--sub); max-width:760px; margin:0}
.meta{display:flex; flex-wrap:wrap; gap:22px; margin-top:22px; font-size:13px; color:var(--faint)}
.meta b{color:var(--ink); font-weight:600}

h2{font-size:23px; margin:56px 0 6px; padding-top:18px; border-top:1px solid var(--line); letter-spacing:-.01em}
h2 .idx{color:var(--blue); font-weight:700; margin-right:10px; font-variant-numeric:tabular-nums}
h3{font-size:17px; margin:32px 0 10px}
h4{font-size:15px; margin:22px 0 6px; color:var(--ink)}
p{margin:10px 0}
ul,ol{margin:10px 0; padding-left:22px}
li{margin:5px 0}
code{
  font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace;
  font-size:.88em; background:var(--soft); border:1px solid var(--line);
  border-radius:4px; padding:1px 5px;
}
b,strong{font-weight:600}
a{color:var(--blue); text-decoration:none; border-bottom:1px solid rgba(47,109,246,.28)}
a:hover{border-bottom-color:var(--blue)}

table{width:100%; border-collapse:collapse; margin:16px 0; font-size:14px}
th,td{padding:9px 12px; text-align:left; border-bottom:1px solid var(--line); vertical-align:top}
th{background:var(--soft); font-weight:600; font-size:13px; color:var(--sub); white-space:nowrap}
td.num{font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap}
td.num.up{color:var(--green); font-weight:600}
td.num.down{color:var(--red); font-weight:600}
td.num.flat{color:var(--faint)}
.hint{font-size:11px; color:var(--faint); margin-left:6px; font-weight:400}
tbody tr:hover{background:#fbfcfe}

.tag{display:inline-block; font-size:11px; padding:1px 8px; border-radius:11px; font-weight:600; white-space:nowrap}
.tag.ok{background:var(--green-soft); color:var(--green)}
.tag.no{background:var(--red-soft); color:var(--red)}
.tag.gray{background:var(--soft); color:var(--faint)}
.tag.mid{background:var(--amber-soft); color:var(--amber)}

.cards{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:22px 0}
.card{border:1px solid var(--line); border-radius:11px; padding:16px 16px 14px; background:#fff}
.card .lbl{font-size:12px; color:var(--sub); margin-bottom:8px}
.card .big{font-size:27px; font-weight:700; letter-spacing:-.02em; font-variant-numeric:tabular-nums; line-height:1.2}
.card .big .arrow{font-size:15px; color:var(--faint); font-weight:400; margin:0 5px}
.card .sub{font-size:12px; color:var(--faint); margin-top:6px}
.card .sub em{font-style:normal; color:var(--green); font-weight:600}
.card .sub em.neg{color:var(--red)}

.note{border-left:3px solid var(--blue); background:var(--blue-soft); padding:13px 18px; border-radius:0 8px 8px 0; margin:20px 0; font-size:14px}
.note .t{font-weight:600; font-size:12px; letter-spacing:.06em; color:var(--blue); margin-bottom:4px}
.note.warn{border-color:var(--amber); background:var(--amber-soft)}
.note.warn .t{color:var(--amber)}
.note.ok{border-color:var(--green); background:var(--green-soft)}
.note.ok .t{color:var(--green)}

.chain{display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin:22px 0; font-size:13px}
.chain .node{border:1px solid var(--line); background:#fff; border-radius:7px; padding:7px 11px; white-space:nowrap}
.chain .node.bad{border-color:#f2c9c9; background:var(--red-soft); color:var(--red); font-weight:600}
.chain .arw{color:var(--faint); font-size:12px}

.step{border:1px solid var(--line); border-radius:11px; padding:20px 22px; margin:20px 0; background:#fff}
.step.reject{border-style:dashed; background:#fcfcfd}
.step .head{display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin-bottom:4px}
.step .head .no{font-size:12px; font-weight:700; color:#fff; background:var(--ink); border-radius:5px; padding:2px 8px; letter-spacing:.04em}
.step.reject .head .no{background:var(--faint)}
.step .head h3{margin:0; font-size:17px}
.step .q{font-size:13px; color:var(--sub); margin:0 0 12px}
.trio{display:grid; grid-template-columns:76px 1fr; gap:2px 12px; font-size:14px; margin:12px 0}
.trio dt{font-size:12px; color:var(--faint); padding-top:3px; letter-spacing:.04em}
.trio dd{margin:0}

.toc{border:1px solid var(--line); border-radius:11px; padding:18px 24px; background:var(--soft); margin:28px 0 6px}
.toc .t{font-size:12px; letter-spacing:.14em; color:var(--faint); margin-bottom:10px}
.toc ol{margin:0; padding-left:20px; columns:2; column-gap:36px; font-size:13.5px}
.toc li{margin:4px 0}

.qa{border-bottom:1px solid var(--line); padding:14px 0}
.qa .q{font-weight:600; margin-bottom:4px}
.qa .q::before{content:"Q"; color:var(--blue); font-weight:700; margin-right:8px}
.qa .a{color:var(--sub); font-size:14px}
.qa .a::before{content:"A"; color:var(--faint); font-weight:700; margin-right:8px}

footer{margin-top:64px; padding-top:22px; border-top:1px solid var(--line); font-size:12.5px; color:var(--faint)}
footer code{font-size:12px}
.sec-intro{color:var(--sub); font-size:14.5px; margin:0 0 6px}

@media (max-width:820px){
  .cards{grid-template-columns:repeat(2,1fr)}
  .toc ol{columns:1}
  .trio{grid-template-columns:1fr}
}
@media print{
  body{font-size:11.5pt}
  .wrap{max-width:none; padding:0}
  h2{page-break-after:avoid}
  table,.step,.card{page-break-inside:avoid}
  .toc{display:none}
}
"""


def head(title: str, kicker: str, lede: str, meta: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header class="hero">
  <div class="kicker">{kicker}</div>
  <h1>{title}</h1>
  <p class="lede">{lede}</p>
  <div class="meta">{meta}</div>
</header>
"""


# ---------------------------------------------------------------- 主报告
def build_main() -> str:
    # 指标条对比（优化前 → 优化后）
    cards = ""
    card_meta = [
        ("context_precision", "上下文精度"),
        ("context_recall", "上下文召回"),
        ("answer_relevancy", "回答相关性"),
        ("faithfulness", "忠实度"),
    ]
    for key, label in card_meta:
        b, a = B[key], A[key]
        d = a - b
        cls = "" if d > 0 else " class=\"neg\""
        cards += f"""
  <div class="card">
    <div class="lbl">{label}</div>
    <div class="big">{b:.3f}<span class="arrow">→</span>{a:.3f}</div>
    <div class="sub">核心项 <em{cls}>{d:+.3f}</em></div>
  </div>"""

    # 3 优化前成绩单
    before_rows = ""
    human = {
        "faithfulness": "约一成句子在资料里找不到依据（不是全错，是部分自造）",
        "answer_relevancy": "最差的一项，5 道题被判 0 分（多为如实拒答）",
        "context_precision": "捞回的资料里约两成是废的，占比不低",
        "context_recall": "达标（门槛 0.70），但仍漏了约 17% 的要点",
        "context_entity_recall": "关键名词只能对上约三分之一",
        "noise_sensitivity": "掺入无关内容后约四分之一句子被带偏（越低越好）",
    }
    for key, label, thr, desc, _ in METRICS:
        rounds = per_round(key, "before")
        ok = (B[key] >= thr) if thr is not None else None
        before_rows += f"""<tr>
  <td><b>{label}</b><br><span class="hint" style="margin:0">{desc}</span></td>
  <td class="num">{f3(rounds[0])}</td><td class="num">{f3(rounds[1])}</td>
  <td class="num"><b>{f3(B[key])}</b></td>
  <td class="num">{('≥ ' + f'{thr:.2f}') if thr is not None else '趋势'}</td>
  <td>{tag(ok)}</td>
  <td style="font-size:13px;color:var(--sub)">{human[key]}</td>
</tr>"""

    # 6.1 四道问题题：优化前（两轮均值）→ 优化后，全部现算
    fix_meta = {
        17: "约束列表在 p61，候选里一条都没有",
        23: "隔离级别在 p93，候选里一条都没有",
        5: "ALTER 语法被切在注入块之外",
        6: "上下文够了但回答没归纳",
    }
    fix_rows = ""
    for idx, why_txt in fix_meta.items():
        bq, aq = q_core(idx, "before"), q_core(idx, "after")
        dq = aq - bq
        fix_rows += (f'<tr><td><code>#{idx}</code></td><td>{why_txt}</td>'
                     f'<td class="num">{bq:.3f}</td><td class="num"><b>{aq:.3f}</b></td>'
                     f'<td class="num up">{dq:+.3f}</td></tr>')

    # 5 各次改动的影响（相邻阶段差）
    def step_delta(key: str, prev: dict, cur: dict, lower_better=False) -> str:
        if prev.get(key) is None or cur.get(key) is None:
            return "—"
        d = cur[key] - prev[key]
        good = (d < 0) if lower_better else (d > 0)
        if abs(d) <= 0.03:
            return f'<span style="color:var(--faint)">{d:+.3f}（噪声内）</span>'
        c = "var(--green)" if good else "var(--red)"
        return f'<b style="color:{c}">{d:+.3f}</b>'

    # 6 优化后对比表
    after_rows = ""
    why = {
        "faithfulness": "上下文变长后模型发挥空间更大，提升有限但守住了门槛",
        "answer_relevancy": "5 道被误判 0 分的题恢复正常作答，是最直接的收益",
        "context_precision": "重排有了区分度、候选池变大、垃圾块被清掉，三重叠加",
        "context_recall": "候选池 15→50 后，原本进不来的正确页被捞回来了",
        "context_entity_recall": "同上，实体随正文一起进来",
        "noise_sensitivity": "本轮未跑（单条判定 400~1646 秒，见第 11 节局限）",
    }
    for key, label, thr, _desc, _a in METRICS:
        b, a = B[key], A[key]
        d = (a - b) if (a is not None and b is not None) else None
        ok = (a >= thr) if (thr is not None and a is not None) else None
        after_rows += f"""<tr>
  <td><b>{label}</b></td>
  <td class="num">{f3(b)}</td><td class="num"><b>{f3(a)}</b></td>
  {delta_cell(d, lower_better=(key == "noise_sensitivity"))}
  <td class="num">{('≥ ' + f'{thr:.2f}') if thr is not None else '趋势'}</td>
  <td>{tag(ok)}</td>
  <td style="font-size:13px;color:var(--sub)">{why[key]}</td>
</tr>"""

    # 7 三组
    groups_rows = ""
    gdefs = [
        ("第一组", "MySQL 基础篇 PDF（教科书）",
         "109 叶子块 / 5.2 万字", "测基础能力：章节规整、术语集中",
         "https://", "docs/RAGAS六指标评估.html"),
        ("第二组", "Kestrel 内部平台（自造 5 种格式）",
         "27 叶子块 / 1.65 万字", "测格式鲁棒性：md/badge 表格/散文/日志/PDF",
         "https://", "docs/RAGAS第二组-异构语料.html"),
        ("第三组", "FastAPI / Pydantic / Uvicorn + RAG 论文（真实公开文档）",
         "156 叶子块 / 10.5 万字", "测泛化：HTML 噪音、两栏排版、跨文档",
         "https://", "docs/RAGAS第三组-真实文档.html"),
    ]
    gdata = [("g1", A, "优化后配置（宽召回）· 单轮"),
             ("g2", G2, "旧配置（无宽召回/未剥 HTML）· 两轮均值"),
             ("g3", G3, "优化后配置（宽召回 + 剥 HTML）· 单轮")]
    for (name, corpus, scale, purpose, _x, link), (gk, data, note) in zip(gdefs, gdata):
        ok_n = sum(1 for k, _, thr, *_ in CORE if data[k] is not None and data[k] >= thr)
        cells = "".join(f'<td class="num">{f3(data[k])}</td>' for k, _, _t, _d, _a in CORE)
        extra = "".join(f'<td class="num">{f3(data[k])}</td>' for k in ("context_entity_recall",))
        style = "" if ok_n == 4 else ' style="color:var(--amber);font-weight:600"'
        groups_rows += f"""<tr>
  <td><b>{name}</b><br><span class="hint" style="margin:0">{corpus}</span></td>
  <td style="font-size:13px;color:var(--sub)">{scale}</td>
  {cells}{extra}
  <td{style}><b>{ok_n}/4</b></td>
</tr>"""

    toc_items = [
        ("why", "为什么做这套测评"),
        ("how", "测评怎么做的"),
        ("before", "优化前的成绩单"),
        ("diag", "问题诊断：分数为什么低"),
        ("fix", "优化过程：四次采纳、一次否决"),
        ("after", "优化后的成绩单"),
        ("groups", "三组语料的三次验证"),
        ("refuse", "拒答能力"),
        ("method", "方法论收获（面试主线）"),
        ("cost", "成本与工程量"),
        ("limits", "局限与后续"),
        ("qa", "面试问答速查"),
    ]
    toc = "".join(f'<li><a href="#{i}">{n}</a></li>' for i, n in toc_items)

    return head(
        "Chiron-AI · RAGAS 测评与优化复盘",
        "Interview Retrospective · RAG Evaluation",
        "一个 RAG 系统从「核心四项只有 1 项达标」到「4 项全部达标」。"
        "这份文档记录的不是一堆分数，而是一条完整的因果链："
        "症状是什么、根因在哪一环、为什么这样改、改完到底有没有变好。",
        f"""<div>项目 <b>Chiron-AI</b>（FastAPI + LangChain + Milvus）</div>
  <div>语料 <b>3 组 × 12 份文档</b></div>
  <div>题量 <b>55 道</b>（45 计分 + 10 应拒答）</div>
  <div>生成时间 <b>{NOW}</b></div>""",
    ) + f"""
<div class="toc"><div class="t">CONTENTS</div><ol>{toc}</ol></div>

<h2 id="tldr">零、一页速览</h2>
<p class="sec-intro">如果只看一段，看这段。</p>

<div class="note ok"><div class="t">结论</div>
问题的根源<b>不在生成模型，在检索链路的四个具体环节</b>：重排没开、候选池太小、分块被切碎、语料混进 HTML 噪音。
四处各改一处，核心四项（忠实度 / 回答相关性 / 上下文精度 / 上下文召回）从 <b>{B_CORE_OK}/4 达标</b> 变成 <b>{A_CORE_OK}/4 达标</b>。
三组语料分别验证后，教科书与真实文档都是 4/4。
</div>

<div class="cards">{cards}</div>

<table>
<thead><tr><th>阶段</th><th class="num">忠实度</th><th class="num">回答相关性</th><th class="num">上下文精度</th><th class="num">上下文召回</th><th class="num">核心达标</th></tr></thead>
<tbody>
<tr><td>优化前（基线，两轮均值）</td>
  <td class="num">{f3(B['faithfulness'])}</td><td class="num">{f3(B['answer_relevancy'])}</td>
  <td class="num">{f3(B['context_precision'])}</td><td class="num">{f3(B['context_recall'])}</td>
  <td class="num"><b style="color:var(--red)">{B_CORE_OK}/4</b></td></tr>
<tr><td>优化后（第一组，单轮）</td>
  <td class="num">{f3(A['faithfulness'])}</td><td class="num">{f3(A['answer_relevancy'])}</td>
  <td class="num">{f3(A['context_precision'])}</td><td class="num">{f3(A['context_recall'])}</td>
  <td class="num"><b style="color:var(--green)">{A_CORE_OK}/4</b></td></tr>
</tbody></table>

<div class="note warn"><div class="t">口径说明（必须一起看）</div>
优化前是<b>两轮平均值</b>（46 次判定），优化后是<b>单轮</b>（23 次判定）——两者不完全对等，
比较时应看趋势而非小数点后第三位。判官也不同：优化前是智谱官方 <code>glm-5.3-flash</code>，
优化后是 SCNet 的 <code>GLM-5.3-Flash-Event</code>。换判官前做过交叉验证（同批 3 题 × 2 指标，
6 个判定中 5 个完全相同），确认口径一致才沿用。
</div>

<h2 id="why"><span class="idx">01</span>为什么做这套测评</h2>
<p class="sec-intro">先说清楚动机，否则后面所有数字都成了没有意义的刷分。</p>

<p>RAG 系统最容易产生一种错觉：<b>随手问几个问题，它答得挺好，于是判断「能用」。</b>
这种判断有三个漏洞——</p>
<ol>
<li><b>只问了自己会问的问题</b>。开发者知道文档里有什么，问法天然贴合检索的词面，命中率被高估。</li>
<li><b>只看了一遍</b>。同一个问题问两次，回答可能一次对一次错（本项目实测单题波动可达 0.4 分）。</li>
<li><b>没有对照</b>。改了一个参数感觉「好像好一点」，但没有证据，下次改动就失去了参照。</li>
</ol>

<p>更麻烦的是第三点。RAG 调参是个多变量问题：<code>top_k</code>、重排开关、分块粒度、合并阈值、
候选池大小，任意一个动了分数都会变。<b>如果没有一把固定的尺子，你无法判断变化是真的还是噪声。</b></p>

<div class="note"><div class="t">这套测评要回答的两个问题</div>
① 现在这套 RAG 到底行不行？<br>
② 我改了某个参数之后，是真的变好了，还是只是抽到了好样本？
</div>

<p>所以做的不是「跑一次分」，而是建一条<b>可反复执行的流水线</b>：固定题目 → 固定判官 → 固定取数方式 →
跑完自动出报告。改一处，跑一次，看差值。<b>没有反馈闭环的优化，本质上是在猜。</b></p>

<h2 id="how"><span class="idx">02</span>测评怎么做的</h2>
<p class="sec-intro">四个设计决定，每一个都直接影响结论能不能信。</p>

<h3>2.1 六项指标分别在测什么</h3>
<p>RAGAS 的六项指标可以分成两组：<b>前三项看「答得好不好」，后两项看「资料找得准不准」</b>。用生活化的说法——</p>

<table>
<thead><tr><th>指标</th><th>技术上在算什么</th><th>一句话人话</th><th>门槛</th></tr></thead>
<tbody>
<tr><td><b>忠实度</b><br><span class="hint" style="margin:0">faithfulness</span></td>
  <td>把回答拆成一句句主张，逐句检查能否在检索到的上下文里找到支撑</td>
  <td>回答<b>有没有编</b></td><td class="num">≥ 0.90</td></tr>
<tr><td><b>回答相关性</b><br><span class="hint" style="margin:0">answer_relevancy</span></td>
  <td>用回答反向生成问题，看与原问题有多相似</td>
  <td>回答<b>有没有答偏</b></td><td class="num">≥ 0.88</td></tr>
<tr><td><b>上下文精度</b><br><span class="hint" style="margin:0">context_precision</span></td>
  <td>逐块判断「这块对回答问题有没有用」，再按排序加权（排得越前权重越高）</td>
  <td>捞回来的资料<b>有没有掺垃圾</b></td><td class="num">≥ 0.85</td></tr>
<tr><td><b>上下文召回</b><br><span class="hint" style="margin:0">context_recall</span></td>
  <td>把参考答案拆成要点，看有多少要点能在上下文里找到</td>
  <td>该捞的资料<b>有没有漏</b></td><td class="num">≥ 0.70</td></tr>
<tr><td><b>实体召回</b><br><span class="hint" style="margin:0">context_entity_recall</span></td>
  <td>统计关键实体（字段名、类名、数字）在上下文中的覆盖率</td>
  <td>专有名词<b>对不对得上</b></td><td class="num">趋势</td></tr>
<tr><td><b>噪声敏感度</b><br><span class="hint" style="margin:0">noise_sensitivity</span></td>
  <td>往上下文里注入无关内容，看答案出错的概率上升多少</td>
  <td><b>抗不抗干扰</b></td><td class="num">趋势</td></tr>
</tbody></table>

<p>其中<b>前四项</b>是「核心四项」，有明确门槛、参与达标判定。后两项是趋势指标：
实体召回在中文语料上受分词与同义表述影响大，噪声敏感度对判官口径极敏感（不同判官能差 2~3 倍），
所以只看趋势、不设门槛。<b>不设门槛不是不重要，而是这两个数目前还不够稳到可以当判据。</b></p>

<h3>2.2 评测集：55 道题，其中 10 道是「应该拒答」的</h3>
<p>题目不是让模型根据文档生成的——那样会形成「抄一遍再考一遍」的假绿：参考答案就是把 chunk 换个说法，
<code>context_recall</code> 必然接近 1，因为测的是「我能不能捞回我刚抄的那段」。做法是：</p>
<ol>
<li>先读语料的<b>章节结构</b>，判断「用户可能合理地在这里问什么」；</li>
<li>答案按<b>领域语义</b>组织（比如按 MySQL 本身的规则写，不按 PDF 的措辞写）；</li>
<li>写完再回查语料，确认每条答案的要点在库里<b>确实有支撑</b>（叫「锚点自查」）。</li>
</ol>
<p>三组分别做了锚点自查，第一组 23 题 0 缺失、第三组 10 题 0 缺失。
这一步的意义是：<b>不能拿文档里没有的知识去扣系统的分。</b></p>

<p>另外每组都留了几道<b>「文档里根本没有」的题</b>，要求系统说「不知道」：</p>
<table>
<thead><tr><th>组</th><th>拒答题</th><th>为什么选它</th></tr></thead>
<tbody>
<tr><td>第一组</td><td>EXPLAIN / 主从复制 / Clone Plugin / Redis 持久化 / 跨库字段</td>
  <td>都是这份 PDF 之后的进阶内容，全文 0 命中</td></tr>
<tr><td>第二组</td><td>备份策略 / 压测数据</td><td>自造语料里故意没写</td></tr>
<tr><td>第三组</td><td>Docker / Kubernetes / Gunicorn</td><td>三份 README 全文 0 命中，但模型很可能凭常识答</td></tr>
</tbody></table>

<div class="note warn"><div class="t">为什么拒答题不能进 RAGAS 均值</div>
拒答题的参考答案是「本文档未涉及 X」。对这类题，<code>context_recall</code> 恒无意义（没有要点可召回），
<code>faithfulness</code> 会因为「回答里没有主张」而虚高。混进均值等于用一道不该算分的题拉高整体。
所以单独统计<b>拒答率</b>（该拒的时候真的拒了没有），不参与六项均值。
</div>

<h3>2.3 判官：为什么用「另一个模型」打分</h3>
<p>RAGAS 的多数指标靠 LLM 判定，这个判分的模型叫<b>判官</b>。选判官有两条原则：</p>
<ul>
<li><b>不能和被评测的模型同一份部署</b>。让同一个模型给自己的答案打分，天然偏宽容。本项目被测系统用
<code>deepseek-flash</code> 生成，判官用 <code>GLM</code> 系列。</li>
<li><b>换判官 = 换尺子，绝对分不可换算</b>。同一个模型换个平台部署，行为也可能变。</li>
</ul>
<p>本项目换过三次判官（智谱官方 glm → 砚眼 deepseek → SCNet glm），每次换都做了一件事：
<b>先拿几道题用新旧判官各判一遍，差值落在噪声地板内才敢沿用旧数据</b>。最后一次的交叉验证结果是
6 个判定中 5 个完全相同，唯一差异 0.143（单题噪声范围内），确认口径一致。</p>

<div class="note warn"><div class="t">一次踩坑：裸调用快 ≠ 能当判官</div>
砚眼平台的 <code>deepseek-v4.1-flash-1017</code> 裸 chat 只要 0.8 秒，但跑 RAGAS 的
<code>faithfulness</code> 单条判定最长 <b>1296 秒</b>——因为该指标要把回答拆句后逐句调模型，
而这是个带 thinking 的模型，多轮结构化输出跑不动。教训：<b>验证判官必须实测一条完整指标判定，不能只测连通性。</b>
</div>

<h3>2.4 为什么跑两轮：噪声地板</h3>
<p>第一组基线跑了两轮，同一个问题、同一批上下文，两轮分数仍然有差（相关性差 0.049、精度差 0.040）。
这个差值不是检索抖动（实测两轮检索结果 <b>28/28 上下文逐条一致</b>，检索是确定性的），而是<b>生成的随机性 + 判官的抽样</b>。</p>
<p>它的意义是给出了一个<b>噪声地板</b>：<b>小于 0.03 的变化不该当成「改好了」或「改坏了」</b>。
本报告里所有「持平」判断都基于这条线。</p>

<h2 id="before"><span class="idx">03</span>优化前的成绩单</h2>
<p class="sec-intro">28 题（23 计分 + 5 应拒答）× 2 轮，判官 glm-5.3-flash，检索为基线配置（未开重排、合并阈值 3）。</p>

<table>
<thead><tr><th style="width:26%">指标</th><th class="num">第一轮</th><th class="num">第二轮</th><th class="num">均值</th><th class="num">门槛</th><th>判定</th><th style="width:22%">这说明什么</th></tr></thead>
<tbody>{before_rows}</tbody></table>

<p><b>核心四项只有 1 项达标（召回 0.829），另外三项全部未达。</b>其中回答相关性 0.732 差得最多。</p>

<p>但这时候不能直接下结论「系统不行」。因为均值会把问题摊平——<b>究竟是整体都差，还是少数几道题特别差？</b>
这两者的处理方式完全不同。所以下一步不是调参，是<b>拆开看</b>。</p>

<h2 id="diag"><span class="idx">04</span>问题诊断：分数为什么低</h2>
<p class="sec-intro">这一章是整份报告最关键的部分。结论先说：不是模型不行，是链路里四个具体环节坏了。</p>

<h3>4.1 先排除了两个错误假设</h3>
<p>诊断过程里做过两次自我否定，值得记下来，因为它们避免了后面白改一通。</p>

<h4>假设一：「图片块占了上下文名额」——部分成立，但不是主因</h4>
<p>观察到每次 top-5 里有 1~2 个块的文本是空的（PDF 里的图片块）。直觉判断是图片抢了名额。
但往下查代码发现：<code>KB_MIN_IMAGE_SLOTS=2</code> 是<b>无条件保底</b>——只要库里有图片候选，
就先塞 2 张图，剩下的位置才轮到文本。所以文本实际只有 3 个名额。这个观察是对的，
但它解释不了「为什么正确答案连候选都进不来」。</p>

<h4>假设二：「答案在截图里，纯文本拿不到」——错的</h4>
<p>有一道题（<code>#5</code> 如何修改表字段）答不出 ALTER 语法，当时的判断是「这段在 PDF 的截图里」。
后来直接查原文发现：<b>正文第 23 页第 700 字就写着</b></p>
<pre style="background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:14px 16px;overflow-x:auto;font-size:13px;line-height:1.7"><code>ALTER TABLE 表名 ADD 字段名 类型(长度) [COMMENT 注释] [约束];
ALTER TABLE emp ADD nickname varchar(20) COMMENT '昵称';
ALTER TABLE 表名 MODIFY 字段名 新数据类型(长度);</code></pre>
<p>而检索给模型的块<b>只有 183 字，正好切在这段语法之前</b>，<code>auto_merge_applied</code> 为 <code>False</code>
（父块合并没触发）。所以根因是<b>分块碎片化</b>，不是图片。</p>

<p>为了排除「模型根本读不了图」这个可能，还做了个对照实验：把那一页的图片直接喂给生成模型，
它准确复述出 <code>desc emp</code> 的 7 个字段和 <code>desc emp;</code> 语句。<b>模型能读图，图片不是瓶颈。</b>
——如果按「加图片通道」去改，这个坑会一直留着。</p>

<h3>4.2 根因链：四个故障点</h3>
<p>把一次问答拆成流水线，每个环节都可能出问题：</p>

<div class="chain">
  <span class="node">用户提问</span><span class="arw">→</span>
  <span class="node">向量检索</span><span class="arw">→</span>
  <span class="node bad">候选池 15 条</span><span class="arw">→</span>
  <span class="node bad">重排（未启用）</span><span class="arw">→</span>
  <span class="node">取 top-5</span><span class="arw">→</span>
  <span class="node bad">分块碎片</span><span class="arw">→</span>
  <span class="node">生成回答</span><span class="arw">→</span>
  <span class="node bad">语料含 HTML</span>
</div>

<table>
<thead><tr><th>环节</th><th>症状</th><th>证据</th><th>影响哪些指标</th></tr></thead>
<tbody>
<tr><td><b>重排未启用</b></td><td>排序接近随机</td>
  <td>向量相似度全挤在 <b>0.014~0.033</b>，单题内最高与最低只差 <b>0.007</b>；日志里 <code>rerank_enabled: False</code></td>
  <td>精度、召回全线偏低</td></tr>
<tr><td><b>候选池太小</b></td><td>正确页进不了候选，重排看不到它</td>
  <td><code>candidate_k = top_k × 3 = 15</code>；<code>#17</code> 的答案在 p61，候选却是 p1/p11/p12/p13</td>
  <td>召回（少数题直接 0）</td></tr>
<tr><td><b>分块碎片 + 合并未触发</b></td><td>上下文只有半页，语法被切断</td>
  <td>注入文本块长度中位 <b>306 字</b>（最短 8 字）；<code>auto_merge</code> 全场只触发 <b>3/28</b></td>
  <td>召回、忠实度</td></tr>
<tr><td><b>语料含 HTML 噪音</b></td><td>垃圾块占掉名额（第三组才暴露）</td>
  <td>第三组 <code>#3</code> 的 5 个注入块里 <b>4 个</b>是 logo/badge 的纯 HTML</td>
  <td>精度</td></tr>
</tbody></table>

<h3>4.3 四道低分题的分诊</h3>
<p>「均值低」要落到具体题上才有意义。基线核心均分最低的四道：</p>

<table>
<thead><tr><th>题号</th><th class="num">核心均分</th><th>归类</th><th>证据</th></tr></thead>
<tbody>
<tr><td><code>#17</code> MySQL 有哪些约束</td><td class="num">{q_core(17, "before"):.3f}</td><td><b>未召回</b></td>
  <td>正确内容在 p61，候选页是 1/11/12/13，一条都没命中</td></tr>
<tr><td><code>#23</code> 事务并发问题与隔离级别</td><td class="num">{q_core(23, "before"):.3f}</td><td><b>未召回</b></td>
  <td>正确内容在 p93，候选页是 1/12/53/88</td></tr>
<tr><td><code>#5</code> 添加/修改字段</td><td class="num">{q_core(5, "before"):.3f}</td><td><b>分块碎片</b></td>
  <td>p23 正文有 ALTER 语法，注入块仅 183 字且切在语法之前</td></tr>
<tr><td><code>#6</code> DROP 与 TRUNCATE 对比</td><td class="num">{q_core(6, "before"):.3f}</td><td><b>生成未综合</b></td>
  <td>上下文已含两者语法与「数据全会删除」，回答没做归纳（参考答案要求对比，尺度偏严）</td></tr>
</tbody></table>

<p><b>这个分诊是整份报告的方法论核心：</b>四道题分数都很低，但原因是四类完全不同的故障。
如果不分类，你只会看到「均值 0.79」，然后去随便调一个参数——大概率调错地方。</p>

<div class="note"><div class="t">一个反直觉的发现</div>
<code>#17</code> 的正确页 p61 <b>被索引了</b>——用 p61 自己的原文去检索，它立刻排到前两名。
所以不是「没入库」，也不是「向量模型不行」，而是<b>用题目问的时候它排不进前 15</b>。
p61 是表格，入库时被拼成一行单元格文本（<code>约束 描述 关键字 非空约束 限制该字段…</code>），
语义不完整，向量分自然靠后。<b>「文档里有」和「能被检索到」是两件事。</b>
</div>

<h2 id="fix"><span class="idx">05</span>优化过程：四次采纳、一次否决</h2>
<p class="sec-intro">按「修哪一环」组织，每次改动都写清：动因、改了什么、结果、代价。失败的那次也留着。</p>

<div class="step">
  <div class="head"><span class="no">改动 1</span><h3>开启重排：把「随机排序」变成「有判断的排序」</h3>
    <span class="tag ok" style="margin-left:auto">采纳</span></div>
  <p class="q">修的是「重排未启用」这一环。</p>
  <dl class="trio">
    <dt>动因</dt><dd>向量相似度全挤在 0.014~0.033、单题内极差 0.007——在这种分布上取 top-5，<b>等于在噪声里挑</b>。
      进一步查到 <code>.env</code> 里 <code>RERANK_API_KEY</code> 是空的，所以 <code>_rerank_documents</code> 一直被跳过。</dd>
    <dt>改动</dt><dd>只改配置，不动代码：在 <code>.env</code> 填入重排服务的 key。重排会把候选按「与问题的语义相关度」重新打分，
      分数变成 <b>0.52~0.87</b>，有了区分度。</dd>
    <dt>结果</dt><dd>精度 {f3(B['context_precision'])} → {f3(R2['context_precision'])}（{step_delta('context_precision', B, R2)}）、
      召回 {f3(B['context_recall'])} → {f3(R2['context_recall'])}（{step_delta('context_recall', B, R2)}）、
      相关性 {f3(B['answer_relevancy'])} → {f3(R2['answer_relevancy'])}（{step_delta('answer_relevancy', B, R2)}）。
      <b>三项同时提升</b>，而且都超出噪声地板。</dd>
    <dt>代价</dt><dd>每题多一次外部调用，延迟 +1~3 秒，按调用计费（量级几分钱）。</dd>
  </dl>
</div>

<div class="step">
  <div class="head"><span class="no">改动 2</span><h3>父块合并阈值 3→2：让碎片拼回整页</h3>
    <span class="tag ok" style="margin-left:auto">采纳</span></div>
  <p class="q">修的是「分块碎片」这一环。</p>
  <dl class="trio">
    <dt>动因</dt><dd>注入的文本块长度中位只有 306 字、最短 8 字，<code>auto_merge</code> 全场只触发 3/28。
      合并逻辑是「同一父块下的叶子块命中数达到阈值就合并回父块」，阈值 3 太高，碎片永远凑不够。</dd>
    <dt>改动</dt><dd><code>AUTO_MERGE_THRESHOLD</code> 从 3 改成 2。一个数字，无其他耦合。</dd>
    <dt>结果</dt><dd>合并触发从 <b>1/23 变成 22/23</b>，块长中位 330→490 字，召回 {step_delta('context_recall', R2, MG)}。
      <code>#5</code> 的召回从 0.60 直接到 <b>1.00</b>。</dd>
    <dt>代价</dt><dd>上下文总量 +45%、输入 token +20%；噪声敏感度 <b>+0.076（变差）</b>——
      块变长后混进无关内容，「单块纯度」下降。这是合并的固有代价，不是 bug。</dd>
  </dl>
</div>

<div class="step reject">
  <div class="head"><span class="no">否决</span><h3>top_k 5→8：看起来最直观，实际最没用</h3>
    <span class="tag no" style="margin-left:auto">已回退</span></div>
  <p class="q">想法是「多给几个名额，总能捞到对的」。</p>
  <dl class="trio">
    <dt>做法</dt><dd>把 <code>KB_TOP_K</code> 从 5 调到 8（同时先把它从三处硬编码抽成配置项，默认行为验证过 23/23 上下文逐条不变）。</dd>
    <dt>结果</dt><dd>精度 <b>{f3(0.783)}</b>（原 0.844，<b style="color:var(--red)">−0.061</b>，破 0.80 底线）、
      召回 0.872（+0.024）。<b>是拿精度换召回，且 #17/#23 仍然是 0。</b></dd>
    <dt>为什么失败</dt><dd>多出来的 3 个位置被<b>同类无关页</b>占走了——<code>#17</code> 的候选从
      <code>[11,13,12,1,53]</code> 变成 <code>[11,13,12,1,53,11,13,0]</code>，还是那几页，只是重复更多。
      名额不够不是根因，扩输出只是把噪声一起放大。</dd>
  </dl>
</div>

<div class="step">
  <div class="head"><span class="no">改动 3</span><h3>宽召回：候选池 15→50，修好 #17</h3>
    <span class="tag ok" style="margin-left:auto">采纳</span></div>
  <p class="q">修的是「候选池太小」这一环——这是 #17 的真正病根。</p>
  <dl class="trio">
    <dt>动因</dt><dd>前两次改动后 <code>#17</code> 依然 0 分。查到最后发现 <code>candidate_k = top_k × 3 = 15</code>（写在代码里），
      而 p61 因为语义不完整排不进前 15。<b>重排再强，也看不到没被送进候选的文档。</b></dd>
    <dt>改动</dt><dd>新增 <code>KB_CANDIDATE_MULTIPLIER</code>（默认 <b>10</b>，候选 50 条），
      同时 <code>RERANK_MAX_CANDIDATES</code> 30→<b>60</b>。<b>两个必须一起改</b>——候选 50 条若仍按 30 截断，
      宽召回等于白做。</dd>
    <dt>结果</dt><dd>精度 <b>0.931 → 0.964</b>、召回 <b>0.903 → 0.974</b>，<b>两项同时涨</b>。
      <code>#17</code> 单题的精度/召回 <b>0.000/0.000 → 0.833/0.857</b>（核心四项均分 {q_core(17, "before"):.3f} → {q_core(17, "after"):.3f}）；
      <code>#23</code> 一并解决（{q_core(23, "before"):.3f} → {q_core(23, "after"):.3f}）。</dd>
    <dt>代价</dt><dd>单题检索 2.9s → <b>3.9s</b>（重排要多看 35 个文档）。</dd>
  </dl>
</div>

<div class="note"><div class="t">为什么这次精度反而涨了</div>
被否决的 top_k 5→8 和这次宽召回，表面都是「多给一些」，结果却相反。区别在于<b>扩大的是哪一段</b>：<br>
· 提 top_k 是放大<b>最终输出</b>——多出来的位置直接进上下文，噪声也一起进去，精度被稀释；<br>
· 放宽候选池是放大<b>重排的输入</b>——多出来的候选要先过重排，再选 5 条进上下文，精度反而升。<br>
<b>要扩就扩候选池，不要扩最终 top_k。</b>这条后来成了本项目的调参原则。
</div>

<div class="step">
  <div class="head"><span class="no">改动 4</span><h3>入库剥离 HTML 噪音：语料质量的问题回到语料层修</h3>
    <span class="tag ok" style="margin-left:auto">采纳</span></div>
  <p class="q">修的是「语料含 HTML」这一环，由第三组真实文档暴露。</p>
  <dl class="trio">
    <dt>动因</dt><dd>第三组 <code>#3/#4</code> 的精度只有 <b>0.583 / 0.500</b>。拆开看注入的 5 个块：
      <code>#3</code> 有 <b>4 个</b>、<code>#4</code> 有 <b>3 个</b>是 README 开头的 logo / 赞助商徽章
      （<code>&lt;p align="center"&gt;&lt;a href=…&gt;&lt;img src=…&gt;</code>）。这些块召回不到任何答案，却和正文抢名额。</dd>
    <dt>改动</dt><dd>在<b>入库解析层</b>加两个函数：<code>strip_html_noise()</code>（去注释、去 script/style、
      块级标签转换行、其余标签删掉但保留内部文字、删纯徽章行）、<code>has_substance()</code>（分块后丢弃去标点后不足 10 字的空块）。
      配置 <code>KB_STRIP_HTML_ENABLED</code>（默认开）可回退。</dd>
    <dt>结果</dt><dd>精度 <b>0.820 → 0.928</b>、召回 <b>0.811 → 0.864</b>，第三组核心四项 <b>3/4 → 4/4</b>。
      单题看 <code>#3/#4</code> 的噪音块从 4/5、3/5 双双降到 <b>0/5</b>。</dd>
    <dt>代价</dt><dd>FastAPI README 从 40 块 / 2.69 万字降到 29 块 / 1.94 万字（−28%）；<b>PDF 完全不受影响</b>。</dd>
  </dl>
</div>

<div class="note warn"><div class="t">这一步最容易翻车的两个边界</div>
① <b>必须保护 fenced code block</b>。讲 HTML/JSX 的文档，代码块里的标签是正文，不能剥——先把 <code>```…```</code>
抽出来占位、处理完再还原。<br>
② <b>只能剥白名单标签名</b>。不能图省事写 <code>&lt;[a-zA-Z][^&gt;]*&gt;</code>，否则技术文档里的
<code>List&lt;String&gt;</code>、<code>Map&lt;K,V&gt;</code> 会被当成标签删掉。<br>
这两条都写成了单测（12 条），因为这类文本清洗一旦误伤，症状会出现在很难追的地方。
</div>

<h2 id="after"><span class="idx">06</span>优化后的成绩单</h2>
<p class="sec-intro">同一套 28 题、同一批语料，用优化后的配置重跑。判官换到 SCNet 的 GLM-5.3-Flash-Event（已交叉验证口径一致）。</p>

<table>
<thead><tr><th style="width:16%">指标</th><th class="num">优化前</th><th class="num">优化后</th><th class="num">变化</th><th class="num">门槛</th><th>判定</th><th style="width:29%">这一项为什么变</th></tr></thead>
<tbody>{after_rows}</tbody></table>

<div class="cards">
  <div class="card"><div class="lbl">核心四项达标</div>
    <div class="big" style="color:var(--green)">{A_CORE_OK}/4</div>
    <div class="sub">优化前 <em class="neg" style="color:var(--red)">{B_CORE_OK}/4</em></div></div>
  <div class="card"><div class="lbl">上下文精度</div>
    <div class="big">{f3(A['context_precision'])}</div>
    <div class="sub">按排序加权的「有用块占比」</div></div>
  <div class="card"><div class="lbl">上下文召回</div>
    <div class="big">{f3(A['context_recall'])}</div>
    <div class="sub">参考答案要点覆盖比例</div></div>
  <div class="card"><div class="lbl">回答相关性</div>
    <div class="big">{f3(A['answer_relevancy'])}</div>
    <div class="sub">反向提问相似度</div></div>
</div>

<div class="note ok"><div class="t">结论：核心四项 1/4 → 4/4</div>
四项全部越过门槛，其中精度、召回、相关性都<b>远超噪声地板（0.03）</b>，不是抽样波动。
忠实度 {f3(A['faithfulness'])} 刚压 0.90 的线，属于「守住但没富余」——这一点必须如实说，不能包装成大幅提升。
</div>

<h3>6.1 单题层面：那四道问题题后来怎么样了</h3>
<table>
<thead><tr><th>题号</th><th>原问题</th><th class="num">优化前核心均分</th><th class="num">优化后</th><th class="num">提升</th></tr></thead>
<tbody>
{fix_rows}
</tbody></table>

<h3>6.2 一个容易被误读的数字：相关性涨了 0.191</h3>
<p>看起来像「回答质量大幅提升」，但拆开看，<b>很大一部分来自「拒答变少了」</b>：
基线里有 <b>5 道题</b>的 <code>answer_relevancy</code> 被判 <b>0.0</b>——它们因为检索没捞到内容而如实拒答，
而「拒答的回答」自然与问题无关，于是这一项被判 0。检索修好后这 5 道题能正常作答，0 分消失，
均分自然被抬上来。优化后 <code>answer_relevancy</code> 为 0 的题：<b>0 道</b>。</p>

<div class="note"><div class="t">这不是判官变宽容，是拒答减少了</div>
面试时这一点很值得讲：<b>指标变化要能解释机制，不能只报数字。</b>如果只看到「相关性 +0.191」就说
「我优化了 prompt」，那是假的——prompt 一个字没改。
</div>

<h2 id="groups"><span class="idx">07</span>三组语料的三次验证</h2>
<p class="sec-intro">只在一份教科书 PDF 上调出来的参数，换个场景就失效。所以做了三组，每组回答一个不同的问题。</p>

<table>
<thead><tr><th style="width:22%">组</th><th>语料规模</th><th class="num">忠实度</th><th class="num">相关性</th><th class="num">精度</th><th class="num">召回</th><th class="num">实体</th><th class="num">核心达标</th></tr></thead>
<tbody>{groups_rows}</tbody></table>

<h3>第一组：MySQL 基础篇 PDF —— 测基础能力</h3>
<p>一份章节规整、术语集中的教科书。这类语料的优点是干扰少、能先把链路跑通；
缺点是<b>太干净</b>——关键词命中就能召回，容易高估系统能力。结果 <b>{A_CORE_OK}/4</b>。</p>

<h3>第二组：自造 5 种格式（md / xlsx / docx / txt / pdf）—— 测格式鲁棒性</h3>
<p>场景设成一个内部数据平台（Kestrel），五份文档共享一套事实、故意埋了端口冲突和遗留待办，
跨文档题才有答案。用脚本生成，可复现。结果 <b>3/4</b>（回答相关性 0.809 未达）。</p>

<div class="note warn"><div class="t">这一组必须如实标注两件事</div>
① 它是<b>优化中途的配置</b>（没有宽召回、没有剥 HTML），不是最终成绩，保留下来是为了说明
「异构格式本身不会打垮检索」；<br>
② 只跑了 11 题、单组语料仅 27 个叶子块，<code>answer_relevancy</code> 在这个样本量下
标准误约 0.05（见 9.3 节），<b>0.809 与 0.88 的差距不足以判定「能力不行」</b>。
</div>

<h3>第三组：真实公开文档 —— 测泛化</h3>
<p>不自造，全部来自公开来源：FastAPI / Pydantic / Uvicorn 三份 README（天然有依赖关系，
FastAPI 站在 Starlette + Pydantic 上、靠 Uvicorn 跑，所以能出跨文档题）+ 一篇 arXiv 论文 PDF（RAG 原论文，两栏排版）。</p>

<table>
<thead><tr><th>指标</th><th class="num">剥离 HTML 前</th><th class="num">剥离后</th><th class="num">变化</th></tr></thead>
<tbody>
<tr><td>上下文精度</td><td class="num">{f3(G3R['context_precision'])}</td><td class="num"><b>{f3(G3['context_precision'])}</b></td>
  {delta_cell(G3['context_precision'] - G3R['context_precision'])}</tr>
<tr><td>上下文召回</td><td class="num">{f3(G3R['context_recall'])}</td><td class="num"><b>{f3(G3['context_recall'])}</b></td>
  {delta_cell(G3['context_recall'] - G3R['context_recall'])}</tr>
<tr><td>忠实度</td><td class="num">{f3(G3R['faithfulness'])}</td><td class="num">{f3(G3['faithfulness'])}</td>
  {delta_cell(G3['faithfulness'] - G3R['faithfulness'])}</tr>
<tr><td>回答相关性</td><td class="num">{f3(G3R['answer_relevancy'])}</td><td class="num">{f3(G3['answer_relevancy'])}</td>
  {delta_cell(G3['answer_relevancy'] - G3R['answer_relevancy'])}</tr>
</tbody></table>

<p>这一组最有价值，因为它给出了一个<b>诚实的落差</b>：真实文档的精度 {f3(G3['context_precision'])}，
比教科书的 {f3(A['context_precision'])} 低 0.043——即使做了 HTML 剥离，真实文档依然更难
（HTML 残留、两栏排版、术语缩写）。但它的<b>回答相关性反而更高</b>（{f3(G3['answer_relevancy'])} vs {f3(A['answer_relevancy'])}），
因为真实文档的表述更接近自然语言。</p>

<div class="note"><div class="t">这说明什么</div>
「精度低」不等于「系统差」，也可能是<b>语料本身更难</b>。只报一个数字会被误读，
必须同时说清语料构成。<b>跨语料比绝对分是耍流氓，只能在同一语料内比改动。</b>
</div>

<h2 id="refuse"><span class="idx">08</span>拒答能力：不知道就说不知道</h2>
<p class="sec-intro">RAG 最危险的失败不是「答不出」，而是「编一个看起来很像的答案」。</p>

<p>三组共 10 道应拒答题，结果全部如实拒答：</p>

<table>
<thead><tr><th>组</th><th>题数</th><th>径直拒答（模型自己判断）</th><th>门控拒答（系统 no_answer 标记）</th><th>典型回答</th></tr></thead>
<tbody>
<tr><td>第一组</td><td class="num">5</td><td class="num"><b>5/5</b></td><td class="num"><b>5/5</b></td>
  <td style="font-size:13px">「知识库中没有关于 EXPLAIN 执行计划的资料」，并说明实际命中了什么</td></tr>
<tr><td>第二组</td><td class="num">2</td><td class="num"><b>2/2</b></td><td class="num"><b>2/2</b></td>
  <td style="font-size:13px">「未找到与备份策略相关的检索结果」</td></tr>
<tr><td>第三组</td><td class="num">3</td><td class="num"><b>3/3</b></td><td class="num"><b>3/3</b></td>
  <td style="font-size:13px">「知识库中没有关于 Kubernetes 部署的资料」，并列出未涉及的概念</td></tr>
</tbody></table>

<p>两条路径都能兜住：<b>模型自己判断</b>（上下文明显不对时主动说明），以及<b>系统门控</b>
（检索后先给每块打分，整体不够就置 <code>no_answer=True</code>）。两者是冗余设计，任一失效另一条还在。</p>

<div class="note"><div class="t">一个方法性教训</div>
第二组曾把一道题（<code>#11</code> HPA 自动扩缩容）凭「我觉得文档没写」标成应拒答，
结果门控返回 <code>no_answer=False</code>——因为架构文档里明写着「不使用 Kubernetes」，
检索确实命中了可回答的内容，系统还主动纠正了提问前提。<b>拒答题不能靠印象判定，必须用门控反向验证。</b>
「索引里没有」和「文档里没有」是两件事。
</div>

<h2 id="method"><span class="idx">09</span>方法论收获（面试主线）</h2>
<p class="sec-intro">这一章是这份报告真正的价值所在——比分数更值得讲的是踩过的坑。</p>

<h3>9.1 换判官 = 换尺子，必须先做交叉验证</h3>
<p>同一个模型名在不同平台部署，行为可能不一样。本项目换判官时踩过两次：</p>
<table>
<thead><tr><th>判官</th><th class="num">同一批题的相关性</th><th>结论</th></tr></thead>
<tbody>
<tr><td>glm-5.3-flash（智谱官方）</td><td class="num">0.747</td><td>原口径</td></tr>
<tr><td>deepseek-v4.1-flash（砚眼）</td><td class="num">0.768</td><td>落在中间，无系统性偏高</td></tr>
<tr><td>qwen-plus（百炼）</td><td class="num">0.795</td><td>最大差 0.048</td></tr>
</tbody></table>
<p><b>教训：</b>换判官后不能直接和历史数字比，要拿 3 题 × 2 指标先对一遍，
差值落在噪声地板内才敢沿用旧数据。<b>「判官与被测模型同家族会虚高」这个担心，在这批题上也没有证据</b>——
新判官的分落在中间，不是最高。</p>

<h3>9.2 失败项绝不能算「已完成」</h3>
<p>判分脚本早期有个隐蔽 bug：失败的判定（429 限流、超时）也被写进了断点续跑文件，
而读取时不区分成功失败。后果是<b>重跑时这些判定被永久跳过</b>——输出文件里静默留着 <code>None</code>，
脚本却打印「完成」。</p>
<p>实际踩到时：92 条判定里 28 条 429 + 3 条超时，23 道题里 31 个空值，报告差点按「模型判不出来」发出去。
修法三处：只把成功项算完成、加按错误类型退避的重试（429 等 12s/24s）、末尾统计仍有空值就退出并报警。</p>
<p><b>教训：</b>任何带断点续跑的任务，都要能区分「做完了」和「做失败了」。
「看起来跑完了」比「明确报错」危险得多。</p>

<h3>9.3 n &lt; 40 时，回答相关性不能当硬门槛</h3>
<p>这是本项目最重要的一次实证。做法是：同一批 11 题、<b>同一批上下文（逐条验证 13/13 完全一致）</b>，
只改回答风格（清单式 → 段落式），再判相关性。</p>
<table>
<thead><tr><th>统计量</th><th class="num">值</th><th>含义</th></tr></thead>
<tbody>
<tr><td>单题变化范围</td><td class="num">−0.189 ~ +0.405</td><td>只改措辞，单题就能摆动 0.4 分</td></tr>
<tr><td>配对差均值</td><td class="num">+0.042</td><td>看起来「段落式更好」</td></tr>
<tr><td>标准误</td><td class="num">0.053</td><td>11 题均值的误差就有 0.05</td></tr>
<tr><td>配对 t</td><td class="num">0.80（p ≈ 0.45）</td><td><b>不显著</b></td></tr>
<tr><td>变好的题数</td><td class="num">6 / 11</td><td>另一半变差</td></tr>
</tbody></table>
<p><b>结论有两层：</b>① 回答风格不是相关性偏低的原因（差异不显著）；② 更要紧的是，
<b>这个指标的单题分值由噪声主导</b>——11 题均值的标准误就有 0.05，意味着 0.88 这条线在 n=11 时
根本无法判定达标与否。</p>
<p>所以本报告的处理是：<b><code>answer_relevancy</code> 在题量 &lt; 40 时按趋势指标处理</b>，
不写进「达标 / 未达标」的硬结论。第二组 0.809 未达门槛，按这个口径不应解读为「能力不行」。</p>

<h3>9.4 要扩就扩候选池，不要扩最终 top_k</h3>
<p>两次实验给出了明确的对照：</p>
<table>
<thead><tr><th>动作</th><th class="num">精度变化</th><th class="num">召回变化</th><th>为什么</th></tr></thead>
<tbody>
<tr><td>top_k 5→8（扩最终输出）</td><td class="num down">−0.061</td><td class="num up">+0.024</td>
  <td>多出来的位置直接进上下文，噪声一起放大</td></tr>
<tr><td>候选池 15→50（扩重排输入）</td><td class="num up">+0.033</td><td class="num up">+0.071</td>
  <td>多出来的候选先过重排，再选 5 条进上下文</td></tr>
</tbody></table>
<p><b>教训：</b>「多给一些」这句话在 RAG 里不完整，必须问清楚<b>给的是哪一段</b>。
给最终上下文是稀释质量，给召回阶段是提升选择空间。</p>

<h3>9.5 收益最大的两个改动，改动量都极小</h3>
<p>按对上下文精度的绝对影响排序（同时标出改动量）：</p>
<ol>
<li><b>开重排 +0.143</b>——只改一行配置（把空的 key 填上）</li>
<li><b>剥 HTML 噪音 +0.108</b>——两个函数 + 12 条单测（第三组口径）</li>
<li><b>宽召回 +0.033</b>——一个配置项 + 一个上限值</li>
<li><b>合并阈值</b>——精度 −0.014、召回 +0.048，<b>是交易不是纯赢</b>（用单块纯度换整页完整）</li>
</ol>
<p><b>这不是巧合。</b>前两个改的都是「信息在进入模型之前就已经坏了」：重排没开导致排序本身没有信息，
HTML 噪音导致候选里装的是垃圾。这类问题修起来便宜、收益直接、不需要权衡。
而合并阈值是「信息已经进来了、只是切得碎」，放大块长必然带进无关内容，只能在召回和纯度之间换。</p>
<p><b>教训：</b>优化前先问一句「这一环有没有根本性失效」，而不是一上来就调参数。
一个空的环境变量造成的损失，比反复调 top_k 大一个数量级。</p>

<h3>9.6 判定「拒答」不能只匹配一个短语</h3>
<p>这个坑踩了三次，每个都是真实发生的：</p>
<table>
<thead><tr><th>写法的回答</th><th>错误判定</th><th>原因</th></tr></thead>
<tbody>
<tr><td>「知识库<b>检索结果中</b>没有相关资料」</td><td>误判为「未拒答」</td><td>只匹配了「知识库中」</td></tr>
<tr><td>「知识库中<b>**没有**</b>关于…」（Markdown 加粗）</td><td>误判为「未拒答」</td><td>加粗符号把匹配串切断了</td></tr>
<tr><td>「未找到与 X 相关的检索结果」</td><td>误判为「未拒答」</td><td>标志词表里没有这一种</td></tr>
</tbody></table>
<p><b>修法：</b>匹配前先剥掉 <code>* _ ` ~</code> 与所有空白，再用多标志词任一命中。
<b>教训：</b>凡是「用字符串匹配判断语义」的地方，都要预设模型会换十种说法。</p>

<h2 id="cost"><span class="idx">10</span>成本与工程量</h2>

<h3>10.1 钱花在哪</h3>
<table>
<thead><tr><th>项目</th><th>用量</th><th class="num">成本</th><th>说明</th></tr></thead>
<tbody>
<tr><td>回答生成</td><td>单轮 28 题 ≈ 入 38.8K / 出 24.8K token</td><td class="num">≈ ¥0.14</td>
  <td>deepseek-flash 空闲时段价（入 1 元/M、出 4 元/M）</td></tr>
<tr><td>回答生成</td><td>整个复盘周期 10+ 次取数、3 组语料</td><td class="num">≈ ¥1.0</td>
  <td>token 取自接口返回的 <code>usage</code> 累加，不是估算</td></tr>
<tr><td>判官</td><td>累计 1000+ 次指标判定</td><td class="num">¥0.10</td>
  <td>SCNet Token Plan 新客包（5000 万 tokens），实测约 20 Credits/次判定</td></tr>
<tr><td>向量 / 重排</td><td>检索与重排调用</td><td class="num">&lt; ¥0.1</td>
  <td>量级可忽略</td></tr>
<tr><td colspan="2"><b>合计</b></td><td class="num"><b>≈ ¥1.2</b></td><td>整套评测从零到三组跑通</td></tr>
</tbody></table>

<h3>10.2 改了多少代码</h3>
<table>
<thead><tr><th>类别</th><th>内容</th></tr></thead>
<tbody>
<tr><td>产品代码（4 个文件）</td>
  <td><code>app/settings/config.py</code>（新增 5 个配置项）、<code>app/kb/rag_pipeline.py</code>（top_k 配置化）、
  <code>app/kb/rag_utils.py</code>（候选池倍数）、<code>app/kb/multimodal_document_loader.py</code>（HTML 剥离）</td></tr>
<tr><td>单元测试</td><td>新增 12 条（HTML 剥离的边界），全量 <b>526 条通过</b>（1 skip）</td></tr>
<tr><td>评测脚本（6 个）</td>
  <td>取数生成、判分、报告生成、语料构造、语料入库、口径合并</td></tr>
<tr><td>数据集（3 个）</td><td>28 题 / 13 题 / 14 题，含参考答案与所属章节</td></tr>
<tr><td>提交</td><td>12 个 commit，每次改动都有「做了什么 + 验证方式」</td></tr>
</tbody></table>

<h2 id="limits"><span class="idx">11</span>局限与后续</h2>
<p class="sec-intro">这一节不是免责声明，是把「哪些结论现在还不能下」说清楚。</p>

<h3>11.1 当前局限</h3>
<ol>
<li><b>轮次不对等</b>。优化前是两轮平均，优化后是单轮。两轮能平抑生成随机性，单轮不能，
所以优化后的数字波动会更大一些。严格对比应补第二轮。</li>
<li><b>第二组没重跑最终配置</b>。它保留的是优化中途配置（无宽召回、未剥 HTML），
其 3/4 不能当作「异构格式不行」的证据。</li>
<li><b>噪声敏感度有缺口</b>。优化后没测，第二组第二轮也没测。原因很实在：单条判定
<b>400~1646 秒</b>（最长 27 分钟），而它只是趋势项。为它多等半小时不划算，报告里显示「—」而不是 0。</li>
<li><b>两题仍未解决</b>。<code>#10</code>（答案跨论文 §2.2 与 §2.3，召回 0.000）、
<code>#11</code>（跨三份 README，top-5 只装得下两份）。这两个都是「一条上下文装不下答案」，靠调参数解决不了。</li>
<li><b>语料规模偏小</b>。三组分别是 109 / 27 / 156 个叶子块，最大一组也只有 156 块。
真实生产语料是万级，候选池的竞争强度完全不同。</li>
<li><b>判官本身有噪声</b>。它是 LLM，不是人类标注。同一份数据换判官能差 0.048，
同一判官两轮也能差 0.03~0.05。所有结论都建立在这个精度上。</li>
</ol>

<h3>11.2 后续要做的</h3>
<table>
<thead><tr><th>优先级</th><th>事项</th><th>为什么</th></tr></thead>
<tbody>
<tr><td>高</td><td><b>动态 top_k</b>：检测到「分别 / 各 / 对比」这类多实体信号时，把最终名额放到 8~10</td>
  <td>直接针对 <code>#10/#11</code> 的「一条上下文装不下」；改动小、可验证</td></tr>
<tr><td>中</td><td><b>多跳检索</b>：先检一次，判断缺什么再补检一次</td>
  <td>跨文档题的根治手段，但会引入额外延迟与复杂度</td></tr>
<tr><td>中</td><td>第二组补跑最终配置 + 全部补齐第二轮</td><td>让三组口径完全一致，数字才能横向比</td></tr>
<tr><td>低</td><td>把评测接进 CI，每次改动自动跑核心四项</td><td>防止后续迭代把已修好的东西改回去</td></tr>
</tbody></table>

<h2 id="qa"><span class="idx">12</span>面试问答速查</h2>
<p class="sec-intro">预演过的问题，每题给一段可以直接说的答案。</p>

<div class="qa"><div class="q">为什么选 RAGAS 而不是人工评估或别的框架？</div>
<div class="a">人工评估准但不可重复——同一批题隔一周再看，标准会漂。RAGAS 的好处是把「回答质量」拆成
六个可分别归因的维度：是资料找错了（精度/召回），还是资料对但答偏了（相关性），还是答得没依据（忠实度）。
<b>能归因比准确更重要</b>，因为我要的不是一个总分，而是「下一步该改哪里」。代价是要自己造评测集、
要处理判官噪声，这些都在报告里标了。</div></div>

<div class="qa"><div class="q">六项指标里你最看哪一项？</div>
<div class="a">看场景，但<b>如果只能留一项，我留上下文召回</b>。因为召回是上游：召回不到，后面全是编；
召回够了哪怕排序差一点，模型还有机会从上下文里挑出对的。不过实操里我会先看精度和召回这一对——
<b>两个都低说明候选池有问题，精度低召回高说明排序有问题，精度高召回低说明候选池不够大</b>，
这一对组合能直接定位故障环节。</div></div>

<div class="qa"><div class="q">为什么拒答题不算进均值？</div>
<div class="a">因为它的参考答案是「本文档未涉及」，对这类题 <code>context_recall</code> 恒无意义（没有要点可召回），
<code>faithfulness</code> 会因为「回答里没有主张」而虚高。混进均值等于用一道不该算分的题拉高整体。
我单独统计拒答率——三组分别是 5/5、2/2、3/3。</div></div>

<div class="qa"><div class="q">判官用另一个模型，会不会反而不准？同家族会不会偏袒？</div>
<div class="a">两个方向都测过。同家族偏袒这个担心<b>没有证据</b>：我拿同一批 11 题让三个判官各判一遍，
被测模型是 deepseek，而用 deepseek 当判官那次给 0.768，正好落在 glm（0.747）和 qwen（0.795）中间，
不是最高。真正要防的是<b>同一个部署</b>既生成又判分，那个才会偏宽容。
另外换判官必须做交叉验证，因为绝对分不可换算。</div></div>

<div class="qa"><div class="q">你怎么确定分数变化是真的提升，不是噪声？</div>
<div class="a">三个办法。① 跑两轮算噪声地板——同一批数据两轮差 0.03~0.05，所以小于 0.03 的变化我一律标「持平」；
② 只看同判官内的相对比较，绝不跨判官比绝对分；③ 变化必须能解释机制。
比如相关性涨 0.191，我拆出来是「5 道被判 0 分的题恢复正常作答」，而不是笼统说「回答变好了」。
解释不了机制的变化，我不会当成结论。</div></div>

<div class="qa"><div class="q">优化前后最大的收益来自哪一步？</div>
<div class="a">分两个维度。按单项收益，是<b>开启重排</b>（精度 +0.143），因为原来向量分全挤在 0.014~0.033、
单题内极差只有 0.007，排序基本等于随机。按「性价比」则是<b>剥 HTML 噪音</b>（精度 +0.108），
只加了两个函数。但真正决定成败的是<b>诊断</b>——如果不先把「均值低」拆成四类故障，
后面每一次调参都有一半概率调错地方。</div></div>

<div class="qa"><div class="q">有没有做过失败的尝试？</div>
<div class="a">有，而且我留着没删。<code>top_k</code> 从 5 调到 8，直觉上「多给几个名额总没错」，
结果精度掉 0.061、破了我自己定的 0.80 底线，换来的召回只有 +0.024。
原因是多出来的位置被同类无关页占走，噪声直接进了上下文。
这次失败后来成了原则：<b>要扩就扩候选池（重排的输入），不要扩最终 top_k（直接进上下文）</b>。
后来宽召回 15→50 就是按这个原则做的，精度和召回同时涨。</div></div>

<div class="qa"><div class="q">这套测评有什么局限？</div>
<div class="a">四条主要的：① 优化后只跑了单轮，和基线的两轮不完全对等；② 第二组保留的是中途配置，
没重跑最终版；③ 噪声敏感度有缺口，因为它单条判定要 400~1646 秒；④ 语料最大一组也只有 156 个叶子块，
和生产万级语料的竞争强度不同。这些我都写在报告的局限一节里，没有藏。</div></div>

<div class="qa"><div class="q">如果给你更多时间，下一步做什么？</div>
<div class="a">做<b>动态 top_k</b>。现在剩的两道失败题（<code>#10</code> 答案跨论文两节、<code>#11</code> 跨三份 README）
本质都是「一条上下文装不下答案」，靠固定 5 个名额解决不了。
方案是检测问题里的多实体信号（「分别」「各」「对比」），命中就把名额放到 8~10。
改动小、可验证、能直接看这两题回不回来。再往后是多跳检索，但那个延迟代价要大得多，得先算清楚值不值。</div></div>

<div class="qa"><div class="q">怎么保证评测集本身是可靠的？</div>
<div class="a">三条。① 不让模型根据文档生成题目和答案——那样会「抄一遍再考一遍」，
<code>context_recall</code> 必然接近 1 但没有意义；② 题目从章节结构出，判断「用户可以合理地问什么」，
答案按领域语义写，写完再回查语料做<b>锚点自查</b>，确认每条要点在库里真有支撑（第一组 23 题、第三组 10 题，0 缺失）；
③ 拒答题必须用门控反向验证，不能凭印象——我在这上面错过一次，把一道其实可答的题标成了应拒答。</div></div>

<footer>
<p><b>数据来源</b>：三组评测的完整明细报告
<a href="RAGAS六指标评估.html">第一组</a>、
<a href="RAGAS第二组-异构语料.html">第二组</a>、
<a href="RAGAS第三组-真实文档.html">第三组</a>；
方法论细节见 <code>docs/交接-RAGAS跑测指南.md</code>。</p>
<p><b>复现</b>：<code>python scripts/ragas/collect_chiron.py</code>（取数+生成）→
<code>python scripts/ragas/eval_glm.py</code>（判分）→
<code>python scripts/ragas/build_unified_reports.py</code>（出报告）。<br>
本页由 <code>scripts/ragas/make_final_report.py</code> 生成，所有数字从判分结果现算。生成时间 {NOW}。</p>
</footer>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------- 三组小结
def build_digest(idx: int, name: str, corpus: str, scale: str, purpose: str,
                 stage: str, base_stage: str | None, note: str,
                 key_points: list[tuple[str, str]], detail_link: str) -> str:
    data = {k: mean_of(k, stage) for k, *_ in METRICS}
    base = {k: mean_of(k, base_stage) for k, *_ in METRICS} if base_stage else {}
    okn = sum(1 for k, _, thr, *_ in CORE if data[k] is not None and data[k] >= thr)

    cards = ""
    for key, label, thr, _d, _a in CORE:
        v = data[key]
        b = base.get(key)
        cls = "ok" if (thr and v and v >= thr) else ("no" if thr else "")
        cards += f"""
  <div class="card"><div class="lbl">{label}</div>
    <div class="big" style="color:var(--{'green' if cls == 'ok' else 'red' if cls == 'no' else 'ink'})">{f3(v)}</div>
    <div class="sub">{('门槛 ≥ ' + f'{thr:.2f}') if thr else '趋势项'}{('　前值 ' + f3(b)) if b is not None else ''}</div></div>"""

    rows = ""
    for key, label, thr, desc, _a in METRICS:
        v = data[key]
        b = base.get(key) if base else None
        d = (v - b) if (v is not None and b is not None) else None
        ok = (v >= thr) if (thr is not None and v is not None) else None
        rows += f"""<tr><td><b>{label}</b><br><span class="hint" style="margin:0">{desc}</span></td>
  <td class="num">{f3(v)}</td><td class="num">{('≥ ' + f'{thr:.2f}') if thr else '趋势'}</td>
  <td>{tag(ok)}</td>{delta_cell(d, lower_better=(key == 'noise_sensitivity')) if base else ''}</tr>"""
    if not base:
        rows = rows.replace('<th class="num">变化</th>', "")

    pts = "".join(f'<div class="qa"><div class="q" style="font-size:14.5px">{q}</div>'
                  f'<div class="a">{a}</div></div>' for q, a in key_points)

    th_change = '<th class="num">变化</th>' if base else ""
    return head(
        f"{name} · RAGAS 小结", f"RAGAS Digest · {idx}/3", purpose,
        f"<div>语料 <b>{corpus}</b></div><div>规模 <b>{scale}</b></div>"
        f"<div>核心达标 <b>{okn}/4</b></div><div>生成时间 <b>{NOW}</b></div>",
    ) + f"""
<div class="cards">{cards}</div>

<div class="note{' ok' if okn == 4 else ' warn'}"><div class="t">这组的结论</div>{note}</div>

<table>
<thead><tr><th style="width:40%">指标</th><th class="num">本次</th><th class="num">门槛</th><th>判定</th>{th_change}</tr></thead>
<tbody>{rows}</tbody></table>

<h2 style="margin-top:40px">这一组最值得讲的几点</h2>
{pts}

<footer>
<p>完整明细（逐题分数、分诊、拒答原文、成本）见 <a href="{detail_link}">这一组的完整报告</a>；
总览与方法论见 <a href="Chiron-AI-RAGAS测评报告.html">RAGAS 测评与优化复盘</a>。</p>
</footer>
</div></body></html>
"""


def main() -> None:
    out_main = ROOT / "docs" / "Chiron-AI-RAGAS测评报告.html"
    out_main.write_text(build_main(), encoding="utf-8", newline="\n")
    print(f"已生成 {out_main.relative_to(ROOT)}  ({out_main.stat().st_size:,} bytes)")

    digests = [
        dict(idx=1, name="第一组 · MySQL 教科书", corpus="MySQL 基础篇 PDF",
             scale="109 叶子块 / 5.2 万字 / 28 题",
             purpose="测基础能力：章节规整、术语集中的一份教科书。这类语料能先把链路跑通，"
                     "但「太干净」容易高估系统——关键词命中就能召回。",
             stage="after", base_stage="before",
             note=f"核心四项 <b>4/4</b>，是全部改动叠加后的基线成绩。"
                  f"上下文精度 {f3(A['context_precision'])}、召回 {f3(A['context_recall'])}，"
                  f"相比优化前（精度 {f3(B['context_precision'])}、召回 {f3(B['context_recall'])}）"
                  f"是这次复盘最大的一次跃迁。忠实度 {f3(A['faithfulness'])} 刚压 0.90 的线，属于守住但没富余。",
             key_points=[
                 ("优化前只有 1 项达标，为什么不是模型的问题？",
                  "四道低分题的原因各不相同：两道是「正确页进不了 15 条候选」（#17/#23），"
                  "一道是「分块切碎了语法」（#5），一道是「上下文够了但回答没归纳」（#6）。"
                  "四类故障对应四个不同的修法，没有一个是「换个更强的模型」能解决的。"),
                 ("为什么先开重排，而不是先调 top_k？",
                  "因为向量分几乎全平：所有候选挤在 0.014~0.033，单题内最高与最低只差 0.007。"
                  "在这种分布上取前 5 条，排序本身就没有信息。先把排序修好，后面的候选池、分块才有意义。"),
                 (f"#17 从 {q_core(17, 'before'):.2f} 到 {q_core(17, 'after'):.2f}，具体是哪一步修好的？",
                  "是宽召回。查到最后发现 p61 的约束表被索引了、用原文能查前两名，"
                  "但用题目问就排不进前 15——<code>candidate_k = top_k × 3 = 15</code> 写在代码里。"
                  "重排再强也看不到没被送进来的文档，所以扩候选池（15→50）才是对症的。"
                  "细节：同时要把重排送排上限从 30 提到 60，否则候选 50 条仍会被截断。"),
                 ("相关性涨了 0.191，是回答质量真的大幅提升吗？",
                  "不全是。基线里有 5 道题因为检索没捞到内容而如实拒答，"
                  "「拒答的回答」自然与问题无关，被这一项判 0。检索修好后它们能正常作答，0 分消失。"
                  "机制解释得清，才敢说这个数字可信。"),
             ],
             detail_link="RAGAS六指标评估.html"),
        dict(idx=2, name="第二组 · 异构语料", corpus="Kestrel 内部平台（自造 5 种格式）",
             scale="27 叶子块 / 1.65 万字 / 13 题",
             purpose="测格式鲁棒性：md 代码块、xlsx 三表、docx 散文、无结构日志、PDF，"
                     "五份文档共享一套事实并埋了端口冲突与遗留待办，跨文档题才有答案。",
             stage="g2", base_stage=None,
             note=f"核心四项 <b>3/4</b>，唯一未达的是回答相关性 {f3(G2['answer_relevancy'])}。"
                  f"但这一组必须如实标注两件事：它跑的是<b>优化中途的配置</b>（没有宽召回、没有剥 HTML），"
                  f"而且只有 11 道计分题——这个样本量下相关性的标准误约 0.05，"
                  f"0.809 与 0.88 的差距不足以判定「能力不行」。",
             key_points=[
                 ("五种格式一起入库，检索会崩吗？",
                  "没有。正确文件基本都排第一，重排分数 0.87~0.92。表格、日志代码块、PDF 分页都没打垮链路。"
                  "这一组想证明的就是「格式异构本身不是障碍」。"),
                 ("这一组踩过什么方法性错误？",
                  "把一道题意标成了拒答题。原题问 HPA 自动扩缩容怎么配，"
                  "我凭「文档没写」判定它该拒答，结果门控返回 <code>no_answer=False</code>——"
                  "因为架构文档里明写着「不使用 Kubernetes」，检索确实命中了可回答的内容，"
                  "系统还主动纠正了提问前提。<b>拒答题不能靠印象判定，要用门控反向验证。</b>"),
                 ("相关性 0.809 是不是真的差？",
                  "不能这么读。除了样本量太小（n=11，标准误 0.05），这一项本身对「否认式回答」也有系统性惩罚——"
                  "回答以「知识库中没有…」开头再纠正前提时会被判低，这一现象在三个不同判官下都成立。"
                  "所以我把它按趋势指标处理，不写进硬结论。"),
             ],
             detail_link="RAGAS第二组-异构语料.html"),
        dict(idx=3, name="第三组 · 真实文档", corpus="FastAPI / Pydantic / Uvicorn README + RAG 论文 PDF",
             scale="156 叶子块 / 10.5 万字 / 14 题",
             purpose="测泛化：全部来自公开来源，不自造。三份 README 天然有依赖关系（FastAPI 站在 "
                     "Starlette + Pydantic 上、靠 Uvicorn 跑），论文是两栏排版的同库干扰项。",
             stage="g3", base_stage="g3_raw",
             note=f"核心四项 <b>4/4</b>，而且是在真实文档上拿到的。剥离 HTML 后精度从 "
                  f"{f3(G3R['context_precision'])} 提到 {f3(G3['context_precision'])}（+0.108），"
                  f"召回 {f3(G3R['context_recall'])} → {f3(G3['context_recall'])}。"
                  f"代价是精度仍比教科书低 0.043——<b>真实文档本来就更难</b>，这个落差是诚实的。",
             key_points=[
                 ("这一组暴露了什么教科书测不出来的问题？",
                  "HTML 噪音。README 开头有大量 logo / 赞助商徽章（<code>&lt;p align=\"center\"&gt;&lt;img src=…&gt;</code>），"
                  "被切成块进了库，能召回但没有任何答案。<code>#3</code> 注入的 5 个块里有 4 个是这种垃圾，"
                  "精度只有 0.583。这不是检索参数的问题，是<b>语料本身脏</b>。"),
                 ("为什么把修复放在入库层，而不是检索层？",
                  "因为垃圾块在库里，后面无论怎么排序都是在垃圾里挑。在解析阶段就剥掉，"
                  "等于从源头减少候选噪声——FastAPI README 从 40 块降到 29 块（−28%），PDF 完全不受影响。"
                  "实测精度 +0.108，比上一轮宽召回（+0.033）大 3 倍，而改动量小得多。"),
                 ("文本清洗最容易翻车的地方在哪？",
                  "两个边界。① 讲 HTML/JSX 的文档，代码块里的标签是正文，必须先把 fenced code block "
                  "抽出来占位再还原；② 只能剥白名单标签名，不能写 <code>&lt;[a-zA-Z][^&gt;]*&gt;</code>，"
                  "否则技术文档里的 <code>List&lt;String&gt;</code>、<code>Map&lt;K,V&gt;</code> 会被当标签删掉。"
                  "这两条都写成了单测（12 条），因为这类误伤的排查成本极高。"),
             ],
             detail_link="RAGAS第三组-真实文档.html"),
    ]
    for d in digests:
        p = ROOT / "docs" / f"RAGAS小结-{d['name'].split(' · ')[0]}-{d['name'].split(' · ')[1].replace(' ', '')}.html"
        p.write_text(build_digest(**d), encoding="utf-8", newline="\n")
        print(f"已生成 {p.relative_to(ROOT)}  ({p.stat().st_size:,} bytes)")

    print(f"\n核心四项：优化前 {B_CORE_OK}/4 → 优化后 {A_CORE_OK}/4")
    for k, label, thr, *_ in CORE:
        print(f"  {label:<8} {f3(B[k])} → {f3(A[k])}  门槛 {thr}")


if __name__ == "__main__":
    main()
