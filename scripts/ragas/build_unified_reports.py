# -*- coding: utf-8 -*-
"""生成两份「透明混口径」报告（2026-11-07 收敛口径）。

为什么不是统一口径：判官切到书生·砚眼 deepseek-v4.1-flash-1017 后，
answer_relevancy 全量重判完成，但 faithfulness 单条最长 1296 秒（RAGAS 拆句逐句判定
+ 模型重试），且服务恢复后给到 420 秒仍 0 结果 —— 该模型不适配 RAGAS 的多轮结构化输出。
用户选择「透明混口径，立刻交付」：不假装统一、不丢弃已跑完的数据，逐项标注判官来源。

数据由 merge_mixed_judges.py 生成，每个数字的来源都可在报告里追溯。

用法：
  python scripts/ragas/merge_mixed_judges.py
  python scripts/ragas/build_unified_reports.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
REPORT = ROOT / "scripts/ragas/make_report.py"

CORE = "faithfulness,answer_relevancy,context_precision,context_recall"

# 换判官后这套说明必须一起改，不要把旧口径留在页面里。
COMMON = [
    f"--metrics={CORE}",
    "--mixed",
    "--metric-sources=answer_relevancy=书生·砚眼 deepseek-v4.1-flash-1017,"
    "faithfulness=glm-5.3-flash,context_precision=glm-5.3-flash,context_recall=glm-5.3-flash",
    "--judge-note=见各指标条下方的逐项标注：本版为混口径报告",
    "--judge=多判官（见各指标条逐项标注）",
    "--judge-cost=混合",
    "--judge-cost-note=砚眼点数 + glm 赠送额度",
    "--judge-cost-detail=回答相关性由书生·砚眼按点数额度判定（未折算人民币），"
    "其余三项来自此前智谱赠送额度，本版未新增费用",
    "--pipeline=固定检索一次 → 基于结果作答（rag_llm=off 基线口径，重排开启）",
    "--embedding=text-embedding-v4",
    "--model=deepseek-flash",
    "--topk=5",
    "--merge=2",
]

SCOPE_NOTE = (
    "⚠ <b>这是透明混口径报告，不是统一口径</b>。换判官后 "
    "<code>answer_relevancy</code> 已用书生·砚眼 <code>deepseek-v4.1-flash-1017</code> "
    "全量重判（两组 × 两轮共 68 次判定，无缺失）；但 <code>faithfulness</code> 在该模型上"
    "单条判定最长 1296 秒、服务恢复后 420 秒仍 0 结果（RAGAS 拆句逐句判定 + 模型重试），"
    "剩余 132 条无法在合理时间内跑完。<b>四个数不能相加、不能平均、不能跨指标比高低</b>，"
    "每条指标条下方都标了自己的判官。<br>"
    "<b>逐项来源</b>：回答相关性 = 砚眼 deepseek-v4.1-flash-1017（本版新判）；"
    "忠实度 / 上下文精度 / 上下文召回 = glm-5.3-flash（此前完整两轮数据，未重判）。<br>"
    "<b>趋势项</b>：实体召回与噪声敏感度不设门槛、换判官后未重判，历史 glm 口径供参考——"
    "第一组实体 0.409 / 0.366、噪声 0.348 / 0.344；第二组实体 0.640、噪声 0.404（单轮），"
    "均不进本版表格。<br>"
    "<b>测量精度提醒</b>：判定实验显示 <code>answer_relevancy</code> 在题量 &lt;40 时"
    "单题噪声极大（仅改措辞单题可摆动 0.4，11 题均值标准误约 0.05），应按趋势指标处理。"
)

# 只有第二组报告需要说明「第二轮复用 / 留空」，放进第一组报告会变成无关信息。
SCOPE_NOTE_G2 = (
    "<b>第二组第二轮的缺口如实列出</b>：该轮是后补取数、智谱额度已耗尽，没有 glm 判分。"
    "其中上下文精度 / 召回的判分入参是 (问题, 参考答案, 检索上下文)，<b>不含回答</b>，"
    "而两轮检索上下文已验证 11/11 逐条一致 → 复用第一轮判定，并在两轮一致性表里标为"
    "「R2 复用 R1」，<b>不计算两轮差</b>（避免把 0 读成极稳定）。"
    "忠实度依赖回答（两轮回答 0/13 相同），不能复用，<b>11 题留空、不编数</b>。"
)

REPORTS = [
    {
        "name": "第一组 · MySQL 基础篇",
        "out": "docs/RAGAS六指标评估.html",
        "args": [
            "--r1=.tmp/ragas/mixed_g1r1.json", "--r2=.tmp/ragas/mixed_g1r2.json",
            "--refuse=.tmp/ragas/bB_r1.json",
            "--gate=.tmp/ragas/b_B_gate.json",
            "--collect1=.tmp/ragas/b_r1.json", "--collect2=.tmp/ragas/b_r2.json",
            "--diagnose=.tmp/ragas/diagnose_b.json",
            "--corpus=MySQL-基础篇.pdf（23 计分 + 5 应拒答）",
            "--scope=u1_a2",
            "--title=Chiron-AI · RAGAS 核心四项（第一组：MySQL 基础篇 · 混口径）",
            "--limits=题目按 PDF 章节结构编写、答案按 MySQL 语义组织，作者仍是助手，需人工复核；"
                    "单份教科书语料的结论不外推到异构文档；#17 仍未召回正确页（p61），属上游召回问题",
        ],
    },
    {
        "name": "第二组 · Kestrel 异构语料",
        "out": "docs/RAGAS第二组-异构语料.html",
        "g2": True,
        "args": [
            "--r1=.tmp/ragas/mixed_g2r1.json", "--r2=.tmp/ragas/mixed_g2r2.json",
            "--borrowed-metrics=context_precision,context_recall",
            "--refuse=.tmp/ragas/k2bB_r1.json",
            "--gate=.tmp/ragas/k2b_B_gate_v2.json",
            "--collect1=.tmp/ragas/k2b_r1.json", "--collect2=.tmp/ragas/k2b_r2.json",
            "--diagnose=.tmp/ragas/diagnose_k2.json",
            "--corpus=Kestrel 9 份异构文档（md / xlsx / docx / txt / pdf，57 chunk；11 计分 + 2 应拒答）",
            "--scope=u1_a3",
            "--title=Chiron-AI · RAGAS 核心四项（第二组：异构语料 · 混口径）",
            "--limits=语料由助手生成、可复现，57 chunk 仍偏小且新增文档是同主题相邻而非近似重复，"
                    "测的是轻量干扰；#11 属否认式回答，answer_relevancy 会系统性判低，已单列",
        ],
    },
]


def check_inputs() -> None:
    """防呆：核心四项有 None 就拒绝出报告，避免"静默完成"的半成品。

    唯一允许的缺口：组2 第二轮的忠实度（依赖回答、不能复用，已在口径说明里明确留空）。
    """
    allowed_none = {"mixed_g2r2": {"faithfulness"}}
    bad = []
    for f in ("mixed_g1r1", "mixed_g1r2", "mixed_g2r1", "mixed_g2r2"):
        p = ROOT / f".tmp/ragas/{f}.json"
        if not p.exists():
            bad.append(f"{f}: 文件不存在（先跑 merge_mixed_judges.py）")
            continue
        rows = json.load(open(p, encoding="utf-8"))
        allowed = allowed_none.get(f, set())
        miss = [(r["idx"], k) for r in rows for k in CORE.split(",")
                if r["scores"].get(k) is None and k not in allowed]
        if miss:
            bad.append(f"{f}: {len(miss)} 个意外缺失（共 {len(rows)} 题），例 {miss[:3]}")
    if bad:
        raise SystemExit("数据不完整，拒绝生成报告：\n  " + "\n  ".join(bad))


def main() -> None:
    check_inputs()
    for spec in REPORTS:
        note = SCOPE_NOTE + ("<br>" + SCOPE_NOTE_G2 if spec.get("g2") else "")
        cmd = [PY, str(REPORT), *COMMON, f"--scope-note={note}",
               *spec["args"], f"--out={spec['out']}"]
        print(f"\n=== 生成 {spec['name']} ===", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("\n两份报告已生成；混口径数据在 .tmp/ragas/mixed_*.json，"
          "来源清单在 .tmp/ragas/mixed_manifest.json")


if __name__ == "__main__":
    main()
