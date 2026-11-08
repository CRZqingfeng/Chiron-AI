# -*- coding: utf-8 -*-
"""生成三组 RAGAS 报告（2026-11-08，宽召回配置 + SCNet GLM 判官）。

判官：三组统一用 SCNet Token Plan 的 GLM-5.3-Flash-Event（第一组由智谱官方 glm 迁过来，
已用 3 题 × 2 指标做过口径交叉验证：6 个判定 5 个完全相同，唯一差异 0.143 落在单题噪声内）。

配置：KB_CANDIDATE_MULTIPLIER=10（候选池 50）、RERANK_MAX_CANDIDATES=60、
KB_TOP_K=5、AUTO_MERGE_THRESHOLD=2、重排开启。宽召回是为了修 #17：
该题答案在 p61，向量分低进不了 15 条候选池，放宽后 precision/recall 从 0.000/0.000
变为 0.833/0.857，全量 23 题精度 0.931→0.964、召回 0.903→0.974（同判官对比）。

轮次：第一、三组单轮；第二组两轮（数据产生于宽召回之前，配置不同，报告里注明）。

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

# 五项：噪声敏感度单条 400~1646 秒（最长 27 分钟），三组都没跑全，不作为本版指标。
FIVE = ("faithfulness,answer_relevancy,context_precision,context_recall,"
        "context_entity_recall")

BASE = [
    f"--metrics={FIVE}",
    "--judge=GLM-5.3-Flash-Event",
    "--judge-note=SCNet Token Plan；换部署前已做 3 题口径交叉验证",
    "--judge-cost=Credits",
    "--judge-cost-note=SCNet Token Plan 套餐（50000 Credits/月）",
    "--judge-cost-detail=判官走 SCNet Token Plan 的 Credits 额度，未折算人民币；"
                         "该模型始终思考，输出侧含 reasoning token",
    "--pipeline=固定检索一次 → 基于结果作答（rag_llm=off 基线口径，重排开启）",
    "--embedding=text-embedding-v4",
    "--model=deepseek-flash",
    "--topk=5",
    "--merge=2",
]

SCOPE_COMMON = (
    "<b>检索配置</b>：候选池 = top_k × 10 = 50 条（宽召回），再由 rerank 从最多 60 条里选最终 5 条；"
    "父块合并阈值 2；重排开启。<br>"
    "<b>为什么放宽候选池</b>：原来 <code>top_k × 3 = 15</code> 条，向量分偏低但内容正确的块"
    "（典型是表格被切成线性文本）进不了候选池，rerank 再好也无从选择——"
    "#17「MySQL 支持哪些约束」命中 p60 而非正文 p61 就是这个原因。放宽后该题 precision/recall "
    "从 <b>0.000/0.000</b> 变为 <b>0.833/0.857</b>，全量 23 题精度 0.931→<b>0.964</b>、"
    "召回 0.903→<b>0.974</b>（同一判官对比），代价是单题检索 2.9s→3.9s。<br>"
    "<b>本版未含噪声敏感度</b>：该指标内部要多轮对照，实测单条 400~1646 秒，"
    "作为趋势项、不设门槛，本版不跑。"
)

REPORTS = [
    {
        "name": "第一组 · MySQL 基础篇（宽召回，单轮）",
        "out": "docs/RAGAS六指标评估.html",
        "scope_extra": "<br><b>轮次</b>：单轮。配置已从「候选 15」改为「候选 50」，"
                       "旧两轮数据基于旧配置，不再与本版并列比较。",
        "args": [
            "--r1=.tmp/ragas/scnet_w1.json",
            "--refuse=.tmp/ragas/wB_new.json",
            "--gate=.tmp/ragas/b_B_gate.json",
            "--collect1=.tmp/ragas/w_new_full.json",
            "--diagnose=.tmp/ragas/diagnose_b.json",
            "--corpus=MySQL-基础篇.pdf（23 计分 + 5 应拒答）",
            "--scope=u1_a2",
            "--title=Chiron-AI · RAGAS 评估（第一组：MySQL 基础篇 · 宽召回）",
            "--limits=题目按 PDF 章节结构编写、答案按 MySQL 语义组织，作者仍是助手，需人工复核；"
                    "单份教科书语料的结论不外推到异构文档",
        ],
    },
    {
        "name": "第二组 · Kestrel 异构语料（两轮，旧配置）",
        "out": "docs/RAGAS第二组-异构语料.html",
        "scope_extra": "<br><b>轮次与配置</b>：两轮，但数据产生于宽召回之前（候选 15），"
                       "与第一、三组配置不同，<b>不要跨组比绝对值</b>。",
        "args": [
            "--r1=.tmp/ragas/scnet_k2r1.json", "--r2=.tmp/ragas/scnet_k2r2.json",
            "--refuse=.tmp/ragas/k2bB_r1.json",
            "--gate=.tmp/ragas/k2b_B_gate_v2.json",
            "--collect1=.tmp/ragas/k2b_r1.json", "--collect2=.tmp/ragas/k2b_r2.json",
            "--diagnose=.tmp/ragas/diagnose_k2.json",
            "--corpus=Kestrel 9 份异构文档（md / xlsx / docx / txt / pdf，57 chunk；11 计分 + 2 应拒答）",
            "--scope=u1_a3",
            "--title=Chiron-AI · RAGAS 评估（第二组：异构语料 · 旧配置两轮）",
            "--limits=语料由助手生成、可复现，57 chunk 偏小且新增文档是同主题相邻而非近似重复；"
                    "#11 属否认式回答，answer_relevancy 会系统性判低，已单列",
        ],
    },
    {
        "name": "第三组 · 真实公开文档（宽召回，单轮）",
        "out": "docs/RAGAS第三组-真实文档.html",
        "scope_extra": "<br><b>轮次</b>：单轮。<b>语料是真实公开文档</b>：三份 GitHub README"
                       "（FastAPI / Pydantic / Uvicorn）+ 一篇 arXiv 论文 PDF（RAG 原论文），"
                       "242 个块 / 约 22.4 万字符，不是自造。FastAPI 的块里混着 HTML 标签、"
                       "论文是两栏排版，这些都是真实文档才有的坑。",
        "args": [
            "--r1=.tmp/ragas/scnet_k3.json",
            "--refuse=.tmp/ragas/k3B_new.json",
            "--collect1=.tmp/ragas/k3_new.json",
            "--corpus=真实公开文档 4 份（FastAPI / Pydantic / Uvicorn README + RAG 论文 PDF，"
                      "242 chunk；11 计分 + 3 应拒答）",
            "--scope=u1_a4",
            "--title=Chiron-AI · RAGAS 评估（第三组：真实公开文档 · 宽召回）",
            "--limits=语料为公开文档快照（2026-11-08 抓取），README 会随上游更新而失效；"
                    "拒答题故意选了 Docker / K8s / Gunicorn——三者在本库 0 命中，但都属于"
                    "「模型可能凭常识答」的题，测的正是该不该拒",
        ],
    },
]


def check_inputs() -> None:
    """防呆：五项任一为 None 就拒绝出报告。"""
    needed = ("scnet_w1", "scnet_k2r1", "scnet_k2r2", "scnet_k3")
    bad = []
    for f in needed:
        p = ROOT / f".tmp/ragas/{f}.json"
        if not p.exists():
            bad.append(f"{f}: 文件不存在")
            continue
        rows = json.load(open(p, encoding="utf-8"))
        miss = [(r["idx"], k) for r in rows for k in FIVE.split(",")
                if r["scores"].get(k) is None]
        if miss:
            bad.append(f"{f}: {len(miss)} 个缺失（共 {len(rows)} 题），例 {miss[:3]}")
    if bad:
        raise SystemExit("数据不完整，拒绝生成报告：\n  " + "\n  ".join(bad))


def main() -> None:
    check_inputs()
    for spec in REPORTS:
        note = SCOPE_COMMON + spec["scope_extra"]
        cmd = [PY, str(REPORT), *BASE, f"--scope-note={note}", *spec["args"],
               f"--out={spec['out']}"]
        print(f"\n=== 生成 {spec['name']} ===", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("\n三份报告已生成。")


if __name__ == "__main__":
    main()
