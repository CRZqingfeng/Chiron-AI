# -*- coding: utf-8 -*-
"""生成两份「各自单一判官」的 RAGAS 报告（2026-11-08 最终口径）。

口径演进到这里才干净：
- 第一组：两轮六指标全部由 glm-5.3-flash（智谱官方 API）判定，此前就已完整，不重判；
- 第二组：两轮六指标全部由 GLM-5.3-Flash-Event（SCNet Token Plan）判定，
  保证同一份报告里没有两把尺子。

两组之间是「同一模型的两次部署」。换平台后用 3 题 × 2 指标做过交叉验证：
6 个判定里 5 个完全相同，唯一差异 0.143 落在单题噪声内，因此第一组旧数据可沿用。
详见 docs/交接-RAGAS跑测指南.md 第十四节。

防呆：任一指标出现 None 就拒绝出报告，不允许再出"半成品"。

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

SIX = ("faithfulness,answer_relevancy,context_precision,context_recall,"
       "context_entity_recall,noise_sensitivity")

# 两组共用的被测系统口径
BASE = [
    f"--metrics={SIX}",
    "--pipeline=固定检索一次 → 基于结果作答（rag_llm=off 基线口径，重排开启）",
    "--embedding=text-embedding-v4",
    "--model=deepseek-flash",
    "--topk=5",
    "--merge=2",
]

SCOPE_G1 = (
    "<b>判官</b>：<code>glm-5.3-flash</code>（智谱官方 API），两轮共 276 次判定，无缺失。<br>"
    "<b>口径</b>：本组两轮数据在切判官之前就已跑完且完整，本次未重判——"
    "同一判官、同一部署，组内两轮与逐题对比都可直接比较。<br>"
    "<b>测量精度提醒</b>：<code>answer_relevancy</code> 在题量 &lt;40 时单题噪声极大"
    "（仅改措辞单题可摆动 0.4，11 题均值标准误约 0.05），宜按趋势指标处理；"
    "实体召回在中文语料下不可靠，只看趋势。"
)

SCOPE_G2 = (
    "<b>判官</b>：<code>GLM-5.3-Flash-Event</code>（SCNet Token Plan），两轮共 132 次判定，无缺失。<br>"
    "<b>为什么换这个名字</b>：Token Plan 的模型 ID 与按量付费不同，"
    "<code>GLM-5.3-Flash</code> 用 Token Plan key 调会返回 403，必须用 "
    "<code>GLM-5.3-Flash-Event</code>；key 前缀是 <code>sk-tp-</code> 而非 <code>sk-</code>。<br>"
    "<b>与第一组的关系</b>：同一模型的两次部署。换平台时用 3 题 × 2 指标做过交叉验证——"
    "6 个判定里 5 个完全相同，唯一差异是 #13 忠实度 0.143，落在单题噪声内"
    "（此前实测 #6 两轮就差 0.38）。因此组间绝对值仍不宜直接比高低（语料不同），"
    "但组内两轮与逐题对比是干净的。<br>"
    "<b>测量精度提醒</b>：同上，<code>answer_relevancy</code> 在题量 &lt;40 时按趋势指标处理。<br>"
    "<b>一处留空</b>：噪声敏感度未跑全——该指标内部要多轮对照，实测单条 400~1646 秒"
    "（最长 27 分钟）。第一轮 9/11 题（index 9、11 两条各卡了 20 分钟以上，放弃），"
    "第二轮整轮未跑。它是趋势项、不设门槛，两轮一致性表里显示「—」，"
    "<b>不是 0、也不是漏数</b>；第一轮的 0.397 是 9 题均值，不与旧数据直接比。"
)

REPORTS = [
    {
        "name": "第一组 · MySQL 基础篇",
        "out": "docs/RAGAS六指标评估.html",
        "judge": "glm-5.3-flash",
        "judge_note": "智谱官方 API；与被测 deepseek 不同家族",
        "judge_cost": "¥ 0",
        "judge_cost_note": "智谱赠送额度（历史数据）",
        "judge_cost_detail": "本组判分为历史数据、此前已产生，本次未新增判官费用",
        "scope": SCOPE_G1,
        "args": [
            "--r1=.tmp/ragas/b_glm_r1.json", "--r2=.tmp/ragas/b_glm_r2.json",
            # 保留历史对比：合并阈值 3→2 之前的两轮，同为 glm，可比
            "--baseline1=.tmp/ragas/rr_glm_r1.json", "--baseline2=.tmp/ragas/rr_glm_r2.json",
            "--change-note=父块合并阈值 3→2（重排保持开启）",
            "--refuse=.tmp/ragas/bB_r1.json",
            "--gate=.tmp/ragas/b_B_gate.json",
            "--collect1=.tmp/ragas/b_r1.json", "--collect2=.tmp/ragas/b_r2.json",
            "--diagnose=.tmp/ragas/diagnose_b.json",
            "--corpus=MySQL-基础篇.pdf（23 计分 + 5 应拒答）",
            "--scope=u1_a2",
            "--title=Chiron-AI · RAGAS 六指标（第一组：MySQL 基础篇）",
            "--limits=题目按 PDF 章节结构编写、答案按 MySQL 语义组织，作者仍是助手，需人工复核；"
                    "单份教科书语料的结论不外推到异构文档；#17 仍未召回正确页（p61），属上游召回问题",
        ],
    },
    {
        "name": "第二组 · Kestrel 异构语料",
        "out": "docs/RAGAS第二组-异构语料.html",
        "judge": "GLM-5.3-Flash-Event",
        "judge_note": "SCNet Token Plan；与第一组同模型不同部署，已做交叉验证",
        "judge_cost": "Credits",
        "judge_cost_note": "SCNet Token Plan 套餐",
        "judge_cost_detail": "判官走 SCNet Token Plan 的 Credits 额度（套餐 50000 Credits/月），"
                             "未折算人民币；该模型始终思考，输出侧含 reasoning token",
        "scope": SCOPE_G2,
        "args": [
            "--r1=.tmp/ragas/scnet_k2r1.json", "--r2=.tmp/ragas/scnet_k2r2.json",
            "--refuse=.tmp/ragas/k2bB_r1.json",
            "--gate=.tmp/ragas/k2b_B_gate_v2.json",
            "--collect1=.tmp/ragas/k2b_r1.json", "--collect2=.tmp/ragas/k2b_r2.json",
            "--diagnose=.tmp/ragas/diagnose_k2.json",
            "--corpus=Kestrel 9 份异构文档（md / xlsx / docx / txt / pdf，57 chunk；11 计分 + 2 应拒答）",
            "--scope=u1_a3",
            "--title=Chiron-AI · RAGAS 六指标（第二组：异构语料）",
            "--limits=语料由助手生成、可复现，57 chunk 仍偏小且新增文档是同主题相邻而非近似重复，"
                    "测的是轻量干扰；#11 属否认式回答，answer_relevancy 会系统性判低，已单列",
        ],
    },
]


def check_inputs() -> None:
    """防呆：六项任一为 None 就拒绝出报告。

    唯一允许的缺口是噪声敏感度：该指标内部要多轮对照，实测单条 400~1646 秒（最长 27 分钟），
    第一轮有 2 条、第二轮整轮都没跑（趋势项、不设门槛，为它多等一小时不划算）。
    报告里会显示「—」，并在口径说明中写明 n 是多少。
    """
    allowed_none = {
        "scnet_k2r1": {"noise_sensitivity"},
        "scnet_k2r2": {"noise_sensitivity"},
    }
    needed = ("b_glm_r1", "b_glm_r2", "scnet_k2r1", "scnet_k2r2")
    bad = []
    for f in needed:
        p = ROOT / f".tmp/ragas/{f}.json"
        if not p.exists():
            bad.append(f"{f}: 文件不存在")
            continue
        rows = json.load(open(p, encoding="utf-8"))
        allow = allowed_none.get(f, set())
        miss = [(r["idx"], k) for r in rows for k in SIX.split(",")
                if r["scores"].get(k) is None and k not in allow]
        if miss:
            bad.append(f"{f}: {len(miss)} 个缺失（共 {len(rows)} 题），例 {miss[:3]}")
    if bad:
        raise SystemExit("数据不完整，拒绝生成报告：\n  " + "\n  ".join(bad))


def main() -> None:
    check_inputs()
    for spec in REPORTS:
        cmd = [
            PY, str(REPORT), *BASE,
            f"--judge={spec['judge']}",
            f"--judge-note={spec['judge_note']}",
            f"--judge-cost={spec['judge_cost']}",
            f"--judge-cost-note={spec['judge_cost_note']}",
            f"--judge-cost-detail={spec['judge_cost_detail']}",
            f"--scope-note={spec['scope']}",
            *spec["args"],
            f"--out={spec['out']}",
        ]
        print(f"\n=== 生成 {spec['name']} ===", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("\n两份报告已生成：每组内部单一判官，不再是混口径。")


if __name__ == "__main__":
    main()
