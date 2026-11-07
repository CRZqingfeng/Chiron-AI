# -*- coding: utf-8 -*-
"""生成两份「统一口径」报告（判官 = 书生·砚眼 deepseek-v4.1-flash-1017）。

为什么单独一个脚本：报告口径有十来个参数，手工敲两次很容易漏一项，
而漏一项就会让报告里出现"智谱免费额度""u1_a2"这类假话。

前置：四批判分已完成（见 docs/交接-RAGAS跑测指南.md 第十二节）。
用法：python scripts/ragas/build_unified_reports.py
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
    "--judge-note=书生·砚眼；与被测同为 deepseek 家族，实测同批 11 题无系统性偏高",
    "--judge-cost=点数",
    "--judge-cost-note=平台点数额度",
    "--judge-cost-detail=判官走书生·砚眼点数额度（输入 1000 点/百万 token、输出 4000 点/百万），未折算人民币",
    "--pipeline=固定检索一次 → 基于结果作答（rag_llm=off 基线口径，重排开启）",
    "--embedding=text-embedding-v4",
    "--model=deepseek-flash",
    "--topk=5",
    "--merge=2",
]

SCOPE_NOTE = (
    "本版只呈现<b>核心四项</b>，两组 × 两轮全部由同一判官 "
    "<code>deepseek-v4.1-flash-1017</code> 判定，因此可以跨组、跨轮比较。<br>"
    "实体召回与噪声敏感度是趋势项、不设门槛，换判官后<b>未重判</b>；"
    "历史 glm-5.3-flash 口径仅作参考：第一组实体 0.409 / 0.366、噪声 0.348 / 0.344；"
    "第二组实体 0.640、噪声 0.404（单轮）。<b>这两项不与核心四项合并判读。</b><br>"
    "另据判定实验：<code>answer_relevancy</code> 在题量 &lt;40 时单题噪声极大"
    "（仅改措辞单题可摆动 0.4，11 题均值标准误约 0.05），应按趋势指标处理，"
    "不宜写进硬性的达标结论。"
)

REPORTS = [
    {
        "name": "第一组 · MySQL 基础篇",
        "out": "docs/RAGAS六指标评估.html",
        "args": [
            "--r1=.tmp/ragas/nj_r1.json", "--r2=.tmp/ragas/nj_r2.json",
            "--refuse=.tmp/ragas/bB_r1.json",
            "--gate=.tmp/ragas/b_B_gate.json",
            "--collect1=.tmp/ragas/b_r1.json", "--collect2=.tmp/ragas/b_r2.json",
            "--diagnose=.tmp/ragas/diagnose_b.json",
            "--corpus=MySQL-基础篇.pdf（23 计分 + 5 应拒答）",
            "--scope=u1_a2",
            "--title=Chiron-AI · RAGAS 核心四项评估（第一组：MySQL 基础篇）",
            "--limits=题目按 PDF 章节结构编写、答案按 MySQL 语义组织，作者仍是助手，需人工复核；"
                    "单份教科书语料的结论不外推到异构文档；#17 仍未召回正确页（p61），属上游召回问题",
        ],
    },
    {
        "name": "第二组 · Kestrel 异构语料",
        "out": "docs/RAGAS第二组-异构语料.html",
        "args": [
            "--r1=.tmp/ragas/nj_k2_r1.json", "--r2=.tmp/ragas/nj_k2_r2.json",
            "--refuse=.tmp/ragas/k2bB_r1.json",
            "--gate=.tmp/ragas/k2b_B_gate_v2.json",
            "--collect1=.tmp/ragas/k2b_r1.json", "--collect2=.tmp/ragas/k2b_r2.json",
            "--diagnose=.tmp/ragas/diagnose_k2.json",
            "--corpus=Kestrel 9 份异构文档（md / xlsx / docx / txt / pdf，57 chunk；11 计分 + 2 应拒答）",
            "--scope=u1_a3",
            "--title=Chiron-AI · RAGAS 核心四项评估（第二组：异构语料）",
            "--limits=语料由助手生成、可复现，57 chunk 仍偏小且新增文档是同主题相邻而非近似重复，"
                    "测的是轻量干扰；#11 属否认式回答，answer_relevancy 会系统性判低，已单列",
        ],
    },
]


def check_inputs() -> None:
    """防呆：核心四项有任一 None 就不许出报告，避免"静默完成"的半成品。"""
    bad = []
    for f in ("nj_r1", "nj_r2", "nj_k2_r1", "nj_k2_r2"):
        p = ROOT / f".tmp/ragas/{f}.json"
        if not p.exists():
            bad.append(f"{f}: 文件不存在")
            continue
        rows = json.load(open(p, encoding="utf-8"))
        miss = sum(1 for r in rows for k in CORE.split(",")
                   if r["scores"].get(k) is None)
        if miss:
            bad.append(f"{f}: {miss} 个 None（共 {len(rows)} 题）")
    if bad:
        raise SystemExit("判分尚未完成，拒绝生成报告：\n  " + "\n  ".join(bad))


def main() -> None:
    check_inputs()
    for spec in REPORTS:
        cmd = [PY, str(REPORT), *COMMON, f"--scope-note={SCOPE_NOTE}",
               *spec["args"], f"--out={spec['out']}"]
        print(f"\n=== 生成 {spec['name']} ===", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("\n两份报告已生成；判分数据在 .tmp/ragas/nj_*.json")


if __name__ == "__main__":
    main()
