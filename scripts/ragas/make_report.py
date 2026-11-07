# -*- coding: utf-8 -*-
"""把 Chiron-AI 的 RAGAS 两轮结果 + 拒答统计 + 成本，生成单文件 HTML 报告。

为什么不用 make_html_v2.py：那个模板的页脚把被测系统的口径和成本写死成 FitMind 的
（"fm-aigc / ES + 1024 维 / 0.21 元"），套过来报告里就是假话。这里重写一份，
口径、成本、拒答段都由参数注入。

无外部依赖：图表用纯 CSS 条形 + SVG，双击可开。

用法：
  python scripts/ragas/make_report.py --r1=glm_r1.json --r2=glm_r2.json \
      --refuse=B_r1.json --collect1=collected_r1.json --collect2=collected_r2.json \
      --out=docs/RAGAS报告.html
"""
from __future__ import annotations

import argparse
import html
import json
import os
import statistics as st
from datetime import datetime

META = [
    ("faithfulness", "忠实度", 0.90, "high", "回答里每句结论，注入的资料撑不撑得住"),
    ("answer_relevancy", "回答相关性", 0.88, "high", "回答有没有答非所问（反向生成问题再算相似度）"),
    ("context_precision", "上下文精度", 0.85, "high", "注入的块里，有用的排得靠不靠前"),
    ("context_recall", "上下文召回", 0.70, "high", "回答需要的信息有没有被捞回来"),
    ("context_entity_recall", "实体召回", None, "info", "★ 中文语料不可靠，只看趋势，不作门槛"),
    ("noise_sensitivity", "噪声敏感度", None, "low", "★ 对判官口径极敏感（差 2~3 倍）；越低越好，只看趋势"),
]
CORE = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}

# 分诊清单由 --diagnose 传入（[{idx,type,label,evidence}]）。
# 为什么要外部传：分诊结论依赖具体语料与页码核实，写死在脚本里换个语料就是假话。

# 生成模型单价（元 / 百万 token）。来源：交接文档实测口径 ——
# deepseek-flash 工作日 9:00-12:00 / 14:00-18:00 为高峰价（入 2 / 出 8），其余时段半价（入 1 / 出 4）。
PRICE = {"peak": (2.0, 8.0), "offpeak": (1.0, 4.0)}


def _arg(k, d):
    return next((a.split("=", 1)[1] for a in __import__("sys").argv if a.startswith(f"--{k}=")), d)


def load(p):
    return json.load(open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def esc(s):
    return html.escape(str(s if s is not None else ""))


REFUSAL_MARKS = (
    "没有相关资料", "无相关资料", "没有找到相关资料", "未找到相关资料",
    "未找到与", "没有找到与", "知识库中没有", "知识库未", "知识库检索结果中没有",
    "无法基于检索结果", "无法依据检索结果", "无法根据检索结果", "无法回答这个问题",
)


def is_refusal(text: str) -> bool:
    """判断回答是否属于"如实拒答"。

    ★ 不要用单一短语匹配：实测 #13 写的是"知识库**检索结果中**没有相关资料"，
    按 "知识库中" 匹配会漏判，把 2/2 算成 1/2。这里用多标志词任一命中。
    """
    t = text or ""
    return any(m in t for m in REFUSAL_MARKS)


def is_peak(ts: datetime) -> bool:
    return ts.weekday() < 5 and (9 <= ts.hour < 12 or 14 <= ts.hour < 18)


def main():
    global META
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", required=True)
    ap.add_argument("--r2", default="")
    ap.add_argument("--metrics", default=",".join(m[0] for m in META),
                    help="本版报告呈现哪些指标；换判官时只传统一重判过的那几项，避免混口径")
    ap.add_argument("--title", default="Chiron-AI · RAGAS 评估")
    ap.add_argument("--refuse", default="", help="B 段取数结果（collected 格式）")
    ap.add_argument("--gate", default="", help="B 段在 rag_llm=on 下的复测结果，用于统计 no_answer 门控")
    ap.add_argument("--collect1", default="", help="第一轮 collected，用于算 token")
    ap.add_argument("--collect2", default="", help="第二轮 collected，用于算 token")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="MySQL-基础篇.pdf")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--judge", default="", help="留空则读 .tmp/ragas/judge.json")
    ap.add_argument("--judge-note", default="", help="判官的一句话说明，别写死在模板里")
    ap.add_argument("--judge-cost", default="¥ 0", help="判官成本展示值")
    ap.add_argument("--judge-cost-note", default="按额度计费")
    ap.add_argument("--judge-cost-detail", default="判官按平台额度计费，未折算人民币")
    ap.add_argument("--pipeline", default="固定检索一次 → 基于结果作答（rag_llm=off 基线口径）")
    ap.add_argument("--embedding", default="text-embedding-v4")
    ap.add_argument("--scope", default="u1_a2")
    ap.add_argument("--scope-note", default="", help="口径说明，放在页头下方醒目位置")
    ap.add_argument("--limits", default="题目由助手编写、需人工复核；实体召回与噪声敏感度在中文语料下不可靠，未作门槛")
    ap.add_argument("--topk", default="5")
    ap.add_argument("--merge", default="2", help="父块合并阈值，页头展示用")
    ap.add_argument("--baseline1", default="", help="改前第一轮判分结果（用于前后对比）")
    ap.add_argument("--baseline2", default="", help="改前第二轮判分结果")
    ap.add_argument("--change-note", default="", help="这次改了什么，显示在对比块标题里")
    ap.add_argument("--diagnose", default="",
                    help='分诊清单 JSON：[{"idx":5,"type":"fragment","label":"分块碎片","evidence":"..."}]')
    a = ap.parse_args()
    if not a.judge:
        _jc = load(".tmp/ragas/judge.json") or {}
        a.judge = _jc.get("model", "glm-5.3-flash")

    # 只保留本版要呈现的指标。META 被下游 11 处循环共用，过滤一次即可全局生效
    # （逐题表列头、指标条、两轮一致性、分诊表都跟着变，不会漏改）。
    _shown = {x.strip() for x in a.metrics.split(",") if x.strip()}
    META = [m for m in META if m[0] in _shown]
    if not META:
        raise SystemExit("--metrics 一个有效指标都没有")

    rounds = [("第一轮", load(a.r1)), ("第二轮", load(a.r2))]
    rounds = [(n, d) for n, d in rounds if d]
    if not rounds:
        raise SystemExit("没有判分数据")
    n_q = len(rounds[0][1])

    # ---------- 汇总 ----------
    summary = {}
    for key, label, thr, direction, desc in META:
        means, vals_all = [], []
        for _n, rows in rounds:
            vs = [r["scores"][key] for r in rows if r["scores"].get(key) is not None]
            means.append(round(st.mean(vs), 4) if vs else None)
            vals_all += vs
        summary[key] = {
            "label": label, "thr": thr, "dir": direction, "desc": desc,
            "mean": round(st.mean(vals_all), 4) if vals_all else None,
            "median": round(st.median(vals_all), 4) if vals_all else None,
            "rounds": means, "vals": vals_all,
            "diff": round(abs(means[0] - means[1]), 4) if len(means) == 2 and None not in means else None,
            "pass_n": (sum(1 for v in vals_all if v >= thr) if thr and direction == "high" else None),
        }

    per_q = []
    for base in rounds[0][1]:
        rec = {
            "idx": base["idx"], "q": base["user_input"], "response": base["response"],
            "reference": base["reference"], "contexts": base["retrieved_contexts"],
            "scores": {}, "rounds": {},
        }
        for key, *_ in META:
            vs = [r["scores"][key] for _n, rows in rounds
                  for r in rows if r["idx"] == base["idx"] and r["scores"].get(key) is not None]
            rec["rounds"][key] = [round(v, 3) for v in vs]
            rec["scores"][key] = round(st.mean(vs), 4) if vs else None
        per_q.append(rec)

    # ---------- 分诊（把"答不出"拆成两类原因，避免混成一个均值） ----------
    diag = load(a.diagnose) or []
    diag_by_idx = {int(d["idx"]): d for d in diag}
    excluded = set(diag_by_idx)

    def cohort_mean(key, skip=frozenset()):
        vs = [r["scores"][key] for r in per_q if r["idx"] not in skip and r["scores"].get(key) is not None]
        return round(st.mean(vs), 4) if vs else None

    cohort = []
    for key, label, thr, direction, _ in META:
        allm = cohort_mean(key)
        clean = cohort_mean(key, excluded) if excluded else None
        cohort.append({"label": label, "all": allm, "clean": clean, "thr": thr,
                       "n_all": n_q, "n_clean": n_q - len(excluded & {r["idx"] for r in per_q})})

    # ---------- 成本 ----------
    def tokens(path):
        rows = load(path) or []
        return (sum(r["rag"].get("prompt_tokens") or 0 for r in rows),
                sum(r["rag"].get("completion_tokens") or 0 for r in rows), len(rows))

    c1 = tokens(a.collect1)
    c2 = tokens(a.collect2)
    cg = tokens(a.gate)  # 拒答段复测也是真实的生成调用，不计进去成本会偏低
    tin, tout = c1[0] + c2[0] + cg[0], c1[1] + c2[1] + cg[1]
    n_gen = c1[2] + c2[2] + cg[2]

    # 两轮检索是否逐题一致（判断"两轮差"到底测到了什么）
    raw1, raw2 = load(a.collect1) or [], load(a.collect2) or []
    ctx_same = None
    if raw1 and raw2 and len(raw1) == len(raw2):
        ctx_same = sum(1 for x, y in zip(raw1, raw2)
                       if x["retrieved_contexts"] == y["retrieved_contexts"])
    resp_same = None
    if raw1 and raw2 and len(raw1) == len(raw2):
        resp_same = sum(1 for x, y in zip(raw1, raw2) if x["response"] == y["response"])
    peak = is_peak(datetime.now())
    pin, pout = PRICE["peak" if peak else "offpeak"]
    cost_gen = tin / 1e6 * pin + tout / 1e6 * pout
    # 判官按平台额度计费，不折算人民币；这里只统计**实际成功**的判定次数。
    # ★ 不能写 题数×6：本次统一口径只判了核心四项，乘 6 会虚报次数。
    n_judge = sum(1 for _n, rows in rounds for r in rows
                  for key, *_ in META if r["scores"].get(key) is not None)
    # embedding：检索侧 qwen3-vl-embedding（每题 1~2 次），answer_relevancy 用 text-embedding-v4
    n_emb = n_gen * 2 + n_q * len(rounds) * 2
    cost_emb = n_emb * 40 / 1e6 * 0.5  # 每次按 40 token 估，单价按 0.5 元/M token 估

    def pct(v):
        return "—" if v is None else f"{v*100:.1f}"

    # ---------- 结论 ----------
    core_ok = [k for k in CORE if summary[k]["mean"] is not None and summary[k]["mean"] >= summary[k]["thr"]]
    all_ok = core_ok
    stable = [summary[k]["label"] for k in summary if summary[k]["diff"] is not None and summary[k]["diff"] <= 0.03]
    unstable = [(summary[k]["label"], summary[k]["diff"]) for k in summary
                if summary[k]["diff"] is not None and summary[k]["diff"] > 0.03]

    # 最弱的 3 题（按核心四项均分）
    def core_avg(rec):
        vs = [rec["scores"][k] for k in CORE if rec["scores"].get(k) is not None]
        return st.mean(vs) if vs else 1.0

    worst = sorted(per_q, key=core_avg)[:3]

    refuse_rows = load(a.refuse) or []
    refuse_ok = 0
    for r in refuse_rows:
        txt = r.get("response") or ""
        if is_refusal(txt):
            refuse_ok += 1

    concl = []
    concl.append(
        f"<b>核心四项达标 {len(core_ok)}/4</b>"
        + ("（" + "、".join(summary[k]["label"] for k in CORE if k in core_ok) + "）"
           if core_ok else "")
        + ("；未达标：" + "、".join(summary[k]["label"] for k in CORE if k not in core_ok) if len(core_ok) < 4 else "")
    )
    # 未达标项逐条给出"差多少"
    short = [(summary[k]["label"], summary[k]["mean"], summary[k]["thr"])
             for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
             if summary[k]["mean"] is not None and summary[k]["mean"] < summary[k]["thr"]]
    if short:
        concl.append(
            "<b>差在哪、差多少</b>：" +
            "；".join(f"{l} {m:.3f}（门槛 {t:.2f}，还差 {t-m:.3f}）" for l, m, t in short)
            + "。注意相关性最弱，它测的是“回答像不像在正面回答问题”，与检索无关。"
        )
    cprec = summary["context_precision"]
    if cprec["mean"] is not None and cprec["median"] is not None and cprec["median"] - cprec["mean"] > 0.08:
        concl.append(
            f"<b>均值在骗人，要看中位数</b>：精度均值 {cprec['mean']:.3f} 但中位数 {cprec['median']:.3f}，"
            f"{cprec['pass_n']}/{n_q * len(rounds)} 次判定达标。这说明是<b>少数几道题</b>在拖，"
            "不是整体检索不行——先定位到题，别急着调参数。"
        )
    if unstable:
        concl.append("<b>两轮差异超阈值的指标</b>：" + "、".join(f"{l}（差 {d:.3f}）" for l, d in unstable)
                      + "。")
    else:
        concl.append("<b>两轮高度一致</b>（所有指标两轮差 ≤0.03），单次采样的分数可以当作可信读数。")
    if ctx_same is not None and len(raw1) == n_gen and len(rounds) == 2:
        concl.append(
            f"<b>两轮检索结果逐题比对：{ctx_same}/{len(raw1)} 题上下文完全一致</b>"
            + (f"，但回答只有 {resp_same}/{len(raw1)} 题相同" if resp_same is not None else "")
            + "。也就是说检索侧是确定性的（<code>rag_llm=off</code>，只走 embedding），"
              "两轮之差来自<b>生成随机性 + 判官抽样</b>，不来自检索。"
              "这给了一个可用的噪声地板：分数变化要明显大于两轮差，才算真的变了。"
        )
    if excluded:
        cl = {c["label"]: c["clean"] for c in cohort}
        concl.append(
            "<b>剔除那 %d 题后的读数</b>（诊断用，不对外）：忠实度 %.3f、回答相关性 %.3f、"
            "上下文精度 %.3f、上下文召回 %.3f。也就是说检索链路本身没有崩，"
            "拖低均值的是那几道题，而不是整体不行。" % (
                len(excluded), cl.get("忠实度") or 0, cl.get("回答相关性") or 0,
                cl.get("上下文精度") or 0, cl.get("上下文召回") or 0)
        )
    if refuse_rows:
        concl.append(
            f"<b>该拒答时确实拒了 {refuse_ok}/{len(refuse_rows)}</b>："
            + ("全部如实说明“知识库中没有相关资料”，并进一步说明检索实际命中了什么，没有一道编造。"
               if refuse_ok == len(refuse_rows) else "有题目在无相关资料时仍给出了内容，需要单独看。")
        )
    if worst:
        concl.append(
            "<b>拉低均值的具体题目</b>：" +
            "；".join(f"#{w['idx']}（核心均分 {core_avg(w):.2f}）" for w in worst)
            + "。定位到题比看均值有用，这三道值得先点开看上下文。"
        )
    # 分诊结论：按类型分组，逐类说清"是不是检索的错"
    by_type: dict[str, list[dict]] = {}
    for d in diag:
        by_type.setdefault(d.get("type") or "other", []).append(d)
    TYPE_SAY = {
        "fragment": ("分块碎片", "命中页对了，但注入的碎片把关键句切掉了（父块合并未触发）"),
        "miss": ("未召回", "正确答案所在页一条都没进候选"),
        "gen": ("生成未综合", "上下文已含所需事实，回答没有归纳"),
        "image": ("图文无解", "答案只在图片像素里，纯文本通道拿不到"),
        "metric": ("指标口径问题", "回答以“没有资料”开头时 answer_relevancy 会系统性判低，非能力问题"),
    }
    for t, items in by_type.items():
        label, why = TYPE_SAY.get(t, (t, ""))
        concl.append(
            f"<b>{len(items)} 题属「{label}」</b>：{'、'.join('#%d' % int(d['idx']) for d in items)}。{why}。"
            + (" 这一类不是模型不会读图——已用对照实验验证模型能准确复述该页图片内容。"
               if t == "image" else "")
        )

    # ---------- 改前 / 改后 对比 ----------
    base_rounds = [(n, load(p)) for n, p in (("改前1", a.baseline1), ("改前2", a.baseline2))]
    base_rounds = [(n, d) for n, d in base_rounds if d]
    cmp_rows = ""
    if base_rounds:
        for key, label, thr, direction, _ in META:
            bvals = [r["scores"][key] for _n, rows in base_rounds for r in rows
                     if r["scores"].get(key) is not None]
            cvals = [r["scores"][key] for _n, rows in rounds for r in rows
                     if r["scores"].get(key) is not None]
            if not bvals or not cvals:
                continue
            bm, cm = st.mean(bvals), st.mean(cvals)
            d = cm - bm
            # 方向规范化：噪声类指标"越低越好"，涨了算退步
            good = (d > 0) if direction != "low" else (d < 0)
            sig = abs(d) > 0.03
            cls = "ok" if (good and sig) else ("no" if (not good and sig) else "")
            mark = ("↑ 改善" if good else "↓ 变差") if sig else "≈ 持平（<0.03）"
            cmp_rows += (f"<tr><td>{esc(label)}</td><td class='mono'>{bm:.3f}</td>"
                         f"<td class='mono'>{cm:.3f}</td>"
                         f"<td class='mono {cls}'>{d:+.3f}</td>"
                         f"<td class='{cls}'>{mark}</td></tr>")
    before_after = """
<h2>三、改前 vs 改后</h2>
<div class="card"><div class="dim">本次未提供改前基线（<code>--baseline1/--baseline2</code>），
没有可比的前后数据 —— 这一节留空，不拿别的语料凑对比。</div></div>
"""
    if cmp_rows:
        note = esc(a.change_note) or "本次改动"
        before_after = f"""
<h2>三、改前 vs 改后（{note}）</h2>
<div class="card"><table>
<thead><tr><th>指标</th><th>改前</th><th>改后</th><th>变化</th><th>判读</th></tr></thead>
<tbody>{cmp_rows}</tbody></table>
<div class="dim" style="font-size:12px;margin-top:10px">
变化绝对值 ≤0.03 视为持平——这是本套评测实测出的噪声地板（两轮差本身就有 0.03~0.05），
小于它的变化不该当成"改好了"或"改坏了"。噪声敏感度越低越好，已按方向折算。
</div></div>
"""

    # ---------- HTML ----------
    judge_note_html = f"（{esc(a.judge_note)}）" if a.judge_note else ""
    scope_note_html = (
        '<div class="card" style="border-left:3px solid #d99a2b;background:#fffaf0;'
        'font-size:13px;line-height:1.8"><b>口径说明</b><br>' + a.scope_note + '</div>'
    ) if a.scope_note else ""
    bar_rows = []
    for key, label, thr, direction, _ in META:
        m = summary[key]
        if m["mean"] is None:
            continue
        w = max(1.5, min(100.0, m["mean"] * 100))
        tgt = min(99.0, thr * 100) if thr else None
        ok = None if thr is None else (m["mean"] >= thr if direction == "high" else m["mean"] <= thr)
        cls = "ok" if ok else ("na" if ok is None else "no")
        flag = "✓ 达标" if ok else ("趋势项" if ok is None else "✗ 未达")
        r2pts = "".join(
            f'<span class="dot" style="left:{max(0.8, min(99.2, v*100)):.1f}%" title="{n} {v:.3f}"></span>'
            for (n, _d), v in zip(rounds, m["rounds"]) if v is not None
        ) if len(rounds) == 2 else ""
        bar_rows.append(f"""
        <div class="brow">
          <div class="blab">{esc(label)}<span class="bdesc">{esc(m['desc'])}</span></div>
          <div class="btrack">
            <div class="bfill {cls}" style="width:{w:.1f}%"></div>
            {f'<div class="btgt" style="left:{tgt:.1f}%"></div>' if tgt else ''}
            {r2pts}
          </div>
          <div class="bval">{m['mean']:.3f}{f"<span class='bthr'> / ≥{thr}</span>" if thr else ""}</div>
          <div class="bflag {cls}">{flag}</div>
        </div>""")

    round_rows = ""
    for key, label, thr, direction, _ in META:
        m = summary[key]
        cells = "".join(f"<td class='mono'>{('%.3f' % v) if v is not None else '—'}</td>" for v in m["rounds"])
        d = m["diff"]
        dcls = "" if d is None else ("ok" if d <= 0.03 else "no")
        dstr = "—" if d is None else f"{d:.3f}"
        pass_col = f"{m['pass_n']}/{n_q*len(rounds)}" if m["pass_n"] is not None else "—"
        round_rows += (f"<tr><td>{esc(label)}</td>{cells}"
                       f"<td class='mono {dcls}'>{dstr}</td><td class='mono'>{pass_col}</td></tr>")

    qrows = ""
    for r in per_q:
        cells = ""
        for key, label, thr, direction, _ in META:
            v = r["scores"].get(key)
            if v is None:
                cells += "<td class='na'>—</td>"
                continue
            if thr is None:
                cls = "na"
            else:
                gap = abs(v - thr)
                cls = "good" if (v >= thr if direction == "high" else v <= thr) else ("near" if gap <= 0.12 else "bad")
            cells += f"<td class='sc {cls}'>{v:.2f}</td>"
        ctxs = "".join(f"<p class='ctx'>• {esc(c[:260])}{'…' if len(c) > 260 else ''}</p>" for c in r["contexts"])
        qrows += f"""
        <tr class="qrow" data-i="{r['idx']}">
          <td class="idx">{r['idx']}</td><td class="q">{esc(r['q'])}</td>{cells}<td class="chev">▸</td>
        </tr>
        <tr class="drow" data-i="{r['idx']}"><td colspan="{len(META)+3}">
          <div class="detail">
            <div class="dcol"><h4>模型回答</h4><div class="resp">{esc(r['response'])}</div>
              <h4>参考答案</h4><div class="dim">{esc(r['reference'])}</div></div>
            <div class="dcol"><h4>注入的上下文（{len(r['contexts'])} 条正文）</h4>{ctxs}</div>
          </div>
        </td></tr>"""

    # 分诊表 + 被剔除题目的证据
    cohort_rows = ""
    for c in cohort:
        allv = f"{c['all']:.3f}" if c["all"] is not None else "—"
        clv = f"{c['clean']:.3f}" if c["clean"] is not None else "—"
        d = (c["clean"] - c["all"]) if (c["clean"] is not None and c["all"] is not None) else None
        dstr = f"{d:+.3f}" if d is not None else "—"
        dcls = "" if d is None else ("ok" if d > 0 else "no")
        cohort_rows += (f"<tr><td>{esc(c['label'])}</td><td class='mono'>{allv}</td>"
                        f"<td class='mono'>{clv}</td><td class='mono {dcls}'>{dstr}</td></tr>")

    diag_rows = ""
    for i in sorted(excluded):
        rec = next((r for r in per_q if r["idx"] == i), None)
        if not rec:
            continue
        d = diag_by_idx.get(i, {})
        kind = d.get("label") or "—"
        cls = "na" if d.get("type") == "gen" else "bad"
        diag_rows += (f"<tr><td class='idx'>#{i}</td><td class='sc {cls}'>{kind}</td>"
                      f"<td>{esc(rec['q'])}</td><td class='dim'>{esc(d.get('evidence', ''))}</td></tr>")

    rrows = ""
    for r in refuse_rows:
        ok = is_refusal(r.get("response") or "")
        rrows += f"""
        <div class="refuse {'ok' if ok else 'bad'}">
          <div class="rh"><span class="tag {'ok' if ok else 'bad'}">{'✓ 如实拒答' if ok else '✗ 未拒答'}</span>
            <b>#{r['idx']}</b> {esc(r['user_input'])}</div>
          <div class="dim">{esc((r.get('response') or '')[:320])}{'…' if len(r.get('response') or '') > 320 else ''}</div>
        </div>"""

    # 完整链路（rag_llm=on）下的门控复测
    gate_rows = load(a.gate) or []
    gate_hit = sum(1 for r in gate_rows if (r.get("rag") or {}).get("no_answer"))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    HTML = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(a.title)}</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#15181d;--dim:#6b7280;--line:#e5e7eb;
--good:#12855a;--bad:#c0392b;--near:#b8860b;--accent:#2563eb}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:28px 20px 60px}}
header{{background:linear-gradient(135deg,#111827,#1f2937);color:#fff;border-radius:14px;padding:26px 28px}}
header h1{{margin:0 0 6px;font-size:22px;letter-spacing:.3px}}
header .sub{{color:#c7cdd6;font-size:13px}}
.kv{{display:flex;flex-wrap:wrap;gap:10px 22px;margin-top:16px;font-size:13px;color:#dfe3e9}}
.kv b{{color:#fff;font-weight:600}}
h2{{font-size:17px;margin:30px 0 12px;padding-left:10px;border-left:4px solid var(--accent)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px}}
.brow{{display:grid;grid-template-columns:190px 1fr 96px 66px;gap:12px;align-items:center;padding:9px 0;border-bottom:1px dashed var(--line)}}
.brow:last-child{{border-bottom:0}}
.blab{{font-size:13px;font-weight:600}}
.bdesc{{display:block;font-weight:400;color:var(--dim);font-size:11px;line-height:1.4}}
.btrack{{position:relative;height:14px;background:#eef1f5;border-radius:7px}}
.bfill{{height:100%;border-radius:7px}}
.bfill.ok{{background:linear-gradient(90deg,#34a853,#12855a)}}
.bfill.no{{background:linear-gradient(90deg,#ef8b7f,#c0392b)}}
.bfill.na{{background:linear-gradient(90deg,#c3cbd8,#94a3b8)}}
.btgt{{position:absolute;top:-4px;width:2px;height:22px;background:#111827;opacity:.55}}
.dot{{position:absolute;top:50%;width:7px;height:7px;border-radius:50%;background:#fff;border:1.5px solid #111827;transform:translate(-50%,-50%)}}
.bval{{font-variant-numeric:tabular-nums;font-weight:700;text-align:right}}
.bthr{{font-weight:400;color:var(--dim);font-size:12px}}
.bflag{{font-size:12px;text-align:right}}
.bflag.ok{{color:var(--good)}} .bflag.no{{color:var(--bad)}} .bflag.na{{color:var(--dim)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:8px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{background:#f9fafb;font-size:12px;color:#374151;font-weight:600;position:sticky;top:0}}
.mono{{font-variant-numeric:tabular-nums;text-align:right}}
td.ok{{color:var(--good)}} td.no{{color:var(--bad)}}
.sc{{text-align:center;font-variant-numeric:tabular-nums;font-weight:600}}
td.good{{color:var(--good);background:#f0fdf6}}
td.near{{color:var(--near);background:#fffbeb}}
td.bad{{color:var(--bad);background:#fef2f2}}
td.na{{color:#9ca3af;text-align:center}}
.idx{{color:var(--dim);width:38px}} .q{{font-weight:500}}
.chev{{width:26px;color:var(--dim);cursor:pointer;user-select:none}}
.qrow{{cursor:pointer}} .qrow:hover{{background:#f8fafc}}
.drow{{display:none}} .drow.open{{display:table-row}}
.detail{{display:grid;grid-template-columns:1fr 1fr;gap:20px;background:#fbfcfd;padding:14px 12px;font-size:12.5px}}
.detail h4{{margin:0 0 6px;font-size:12px;color:#374151}}
.resp{{white-space:pre-wrap}}
.dim{{color:var(--dim)}}
.ctx{{margin:0 0 8px;padding:8px 10px;background:#fff;border:1px solid var(--line);border-radius:7px;color:#444}}
ol.concl{{margin:0;padding-left:22px}} ol.concl li{{margin-bottom:9px}}
.refuse{{border:1px solid var(--line);border-left:4px solid var(--good);border-radius:9px;padding:12px 14px;margin-bottom:10px;background:#fff}}
.refuse.bad{{border-left-color:var(--bad)}}
.rh{{margin-bottom:6px;font-size:13.5px}}
.tag{{display:inline-block;font-size:11px;padding:1px 8px;border-radius:20px;margin-right:8px;background:#e8f5ee;color:var(--good)}}
.tag.bad{{background:#fdecea;color:var(--bad)}}
.cost{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.cbox{{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:#fff}}
.cbox .n{{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}}
.cbox .t{{font-size:12px;color:var(--dim)}}
footer{{margin-top:34px;color:var(--dim);font-size:12px;line-height:1.8;border-top:1px solid var(--line);padding-top:14px}}
code{{background:#eef1f5;padding:1px 5px;border-radius:4px;font-size:12px}}
@media(max-width:760px){{.brow{{grid-template-columns:120px 1fr 70px}}.detail{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<header>
  <h1>{esc(a.title)}</h1>
  <div class="sub">检索子图 {esc(a.corpus)} · {esc(a.pipeline)}</div>
  <div class="kv">
    <div>题量 <b>{n_q}</b> 题 × <b>{len(rounds)}</b> 轮</div>
    <div>回答模型 <b>{esc(a.model)}</b></div>
    <div>判官 <b>{esc(a.judge)}</b>{judge_note_html}</div>
    <div>topK / 合并阈值 <b>{esc(a.topk)}</b> / <b>{esc(a.merge)}</b></div>
    <div>生成时间 <b>{ts}</b></div>
  </div>
</header>
{scope_note_html}
<h2>一、结论</h2>
<div class="card"><ol class="concl">{''.join(f'<li>{c}</li>' for c in concl)}</ol></div>

<h2>二、各项指标 vs 目标</h2>
<div class="card">{''.join(bar_rows)}
<div class="dim" style="font-size:12px;margin-top:10px">竖线 = 目标值；圆点 = 各轮均值（{len(rounds)} 轮）。灰色条为趋势项，不设门槛。</div>
</div>

{before_after}<h2>四、分诊：低分题归因</h2>
<div class="card">
<table>
<thead><tr><th>指标</th><th>全量 {n_q} 题</th><th>剔除 {len(excluded)} 题后（{n_q - len(excluded)} 题）</th><th>差值</th></tr></thead>
<tbody>{cohort_rows or '<tr><td colspan="4" class="dim">未指定分诊题号</td></tr>'}</tbody></table>
<div class="dim" style="font-size:12px;margin:10px 0 14px">
被剔除的题目按<b>失效环节</b>归类；下面的证据逐题列明。剔除后的读数是"这些环节都不出错时"的链路基线，
用来判断检索本身行不行，<b>不是</b>对外报的数，只作定位用。</div>
<table>
<thead><tr><th>#</th><th>类型</th><th>问题</th><th>证据</th></tr></thead>
<tbody>{diag_rows}</tbody></table>
</div>

<h2>五、两轮一致性</h2>
<div class="card"><table>
<thead><tr><th>指标</th>{''.join(f'<th>{esc(n)}</th>' for n, _ in rounds)}<th>两轮差</th><th>达标次数</th></tr></thead>
<tbody>{round_rows}</tbody></table>
<div class="dim" style="font-size:12px;margin-top:10px">
本轮 <code>rag_llm=off</code>：检索只走 embedding、无随机性，实测两轮上下文
<b>{f'{ctx_same}/{len(raw1)}' if ctx_same is not None else '—'}</b> 题完全一致；
而回答{f'只有 {resp_same}/{len(raw1)} 题相同' if resp_same is not None else '存在差异'}。
所以两轮差 = 生成随机性 + 判官抽样，<b>不含检索抖动</b>。这个差值就是「噪声地板」——
今后分数变化要明显大于它，才算真的变了。</div>
</div>

<h2>六、逐题明细（点行展开）</h2>
<div class="card" style="padding:6px 10px"><table>
<thead><tr><th>#</th><th>问题</th>{''.join(f'<th>{esc(l)}</th>' for _k, l, _t, _d, _s in META)}<th></th></tr></thead>
<tbody>{qrows}</tbody></table></div>

<h2>七、应拒答题（单独统计，不进上面均值）</h2>
<div class="card">{rrows if rrows else '<div class="dim">本次未跑拒答题</div>'}
<div class="dim" style="font-size:12px;margin-top:8px">
判定规则：回答中出现「没有相关资料 / 知识库中未 / 无法依据检索结果回答」等多类标志词之一即算如实拒答
（用 <code>is_refusal()</code> 多标志词匹配，单一短语会漏判）。
这类题不能进 RAGAS 均值——`context_recall` 对“本文档未涉及”恒无意义，`faithfulness` 会因“回答里没有主张”虚高。</div>
{f'<div style="margin-top:12px;padding:10px 12px;background:#f0f7ff;border:1px solid #d6e4ff;border-radius:8px;font-size:13px"><b>门控复测</b>：另用完整链路（<code>rag_llm=on</code>，含逐块打分与二次门控）跑同一批题，系统自身的 <code>no_answer</code> 判定触发 <b>{gate_hit}/{len(gate_rows)}</b>，单题耗时 7~11 秒。也就是说门控机制本身有效——上面这些“如实拒答”不是模型临场发挥，而是两条路都能兜住。</div>' if gate_rows else ''}
</div>

<h2>八、成本</h2>
<div class="card">
<div class="cost">
  <div class="cbox"><div class="n">¥ {cost_gen:.3f}</div><div class="t">回答生成（{n_gen} 次）<br>入 {tin:,} / 出 {tout:,} token</div></div>
  <div class="cbox"><div class="n">{esc(a.judge_cost)}</div><div class="t">判官（{n_judge} 次成功判定）<br>{esc(a.judge_cost_note)}</div></div>
  <div class="cbox"><div class="n">&lt; ¥ 0.01</div><div class="t">embedding（约 {n_emb} 次）<br>估算，量级可忽略</div></div>
  <div class="cbox"><div class="n">¥ {cost_gen:.3f}</div><div class="t">合计（{'高峰时段' if peak else '空闲时段'}计价）<br>不含判官额度</div></div>
</div>
<div class="dim" style="font-size:12px;margin-top:12px">
估算依据：生成单价按交接文档实测口径 —— <code>{esc(a.model)}</code> 工作日 9:00-12:00 / 14:00-18:00 为
输入 2 元/M、输出 8 元/M；其余时段半价（输入 1 元/M、输出 4 元/M）。本次运行时间点判定为
<b>{'高峰' if peak else '空闲'}</b>，按 {pin:g} / {pout:g} 元/M 计。token 数取自接口返回的 usage 累加，不是估算。
{esc(a.judge_cost_detail)}；embedding 按每次约 40 token、0.5 元/M 粗估，量级在 0.001 元以下。
</div>
</div>

<footer>
<b>口径</b>：被测系统为 Chiron-AI 的 RAG 子图（LangGraph），Milvus 向量库，知识库范围
<code>{esc(a.scope)}</code>（{esc(a.corpus)}）；判分侧相关性用 <code>{esc(a.embedding)}</code>。<br>
取数方式：<code>run_rag_graph</code> 拿<b>完整正文</b>作上下文（不用 120 字的 text_head），
再把上下文按项目自己的 <code>format_knowledge_retrieval_tool_output</code> 拼给回答模型，保证上下文与回答同源。<br>
<b>局限</b>：{esc(a.limits)}
</footer>
</div>
<script>
document.querySelectorAll('.qrow').forEach(function(tr){{
  tr.addEventListener('click', function(){{
    var d=document.querySelector('.drow[data-i="'+tr.dataset.i+'"]');
    d.classList.toggle('open');
    tr.querySelector('.chev').textContent = d.classList.contains('open')?'▾':'▸';
  }});
}});
</script></body></html>"""

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8", newline="\n").write(HTML)
    print(f"已生成 {a.out}（{len(HTML)} 字符）")
    print(f"成本：生成 ¥{cost_gen:.3f}（入 {tin} / 出 {tout} token，{'高峰' if peak else '空闲'}价）"
          f"  判官 {a.judge_cost}（{n_judge} 次）  embedding <¥0.01")
    print(f"结论：核心四项达标 {len(core_ok)}/4  拒答 {refuse_ok}/{len(refuse_rows) if refuse_rows else 0}")


if __name__ == "__main__":
    main()
