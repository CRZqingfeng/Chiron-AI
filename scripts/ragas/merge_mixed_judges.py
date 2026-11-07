# -*- coding: utf-8 -*-
"""生成「透明混口径」数据：相关性用新判官，其余三项用 glm 旧判官。

背景（2026-11-07）：判官切到书生·砚眼 deepseek-v4.1-flash-1017 后，
answer_relevancy 全量重判完成，但 faithfulness 单条最长 1296 秒（RAGAS 拆句逐句判定
+ 模型重试），剩余 132 条预计 3 小时以上。用户选择「透明混口径，立刻交付」：
不假装统一、也不丢弃已跑完的数据，逐项标注判官来源。

数据拼接规则（每一条都可在报告里追溯）：
- answer_relevancy：nj_*.partial.jsonl 里成功的判定，四批全量齐。
- faithfulness / context_precision / context_recall：glm-5.3-flash 的完整结果。
- 组2 第二轮没有 glm 判分（取数是后补的，智谱额度已耗尽）：
  · context_precision / context_recall 的 ascore 入参是
    (user_input, reference, retrieved_contexts)，**不含 response**，
    而两轮检索上下文已验证 13/13 逐条一致 → 复用 R1 判定（标注为「复用」）。
    脚本会先断言这个前提，不成立就报错，不允许悄悄复用。
  · faithfulness 依赖 response（两轮回答 0/13 相同）→ 留 None，报告里明确留空，不编数。

用法：python scripts/ragas/merge_mixed_judges.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / ".tmp/ragas"

NEW_JUDGE = "deepseek-v4.1-flash-1017"
OLD_JUDGE = "glm-5.3-flash"
CORE = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# (输出名, 取数基线, 新判官 partial, glm 结果, 复用来源)
SPEC = [
    ("mixed_g1r1", "bA_r1.json", "nj_r1.partial.jsonl", "b_glm_r1.json", None),
    ("mixed_g1r2", "bA_r2.json", "nj_r2.partial.jsonl", "b_glm_r2.json", None),
    ("mixed_g2r1", "k2bA_r1.json", "nj_k2_r1.partial.jsonl", "k2b_glm_r1.json", None),
    ("mixed_g2r2", "k2bA_r2.json", "nj_k2_r2.partial.jsonl", None, "k2b_glm_r1.json"),
]

KEEP = ("idx", "user_input", "response", "reference",
        "retrieved_contexts", "source_meta", "rag")


def load(name: str):
    return json.load(open(TMP / name, encoding="utf-8"))


def new_relevancy(partial: str) -> dict[int, float]:
    """从 partial 里取 answer_relevancy 的成功值（失败行会被后写成功行覆盖）。"""
    out = {}
    for line in open(TMP / partial, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        r = d.get("result") or {}
        if d["metric"] == "answer_relevancy" and not r.get("error") and r.get("score") is not None:
            out[int(d["idx"])] = r["score"]
    return out


def glm_scores(name: str) -> dict[int, dict]:
    return {int(r["idx"]): r["scores"] for r in load(name)}


def main() -> None:
    # 复用前先验证：组2 两轮的检索上下文必须逐条一致，否则不能复用 R1 的 context 类指标
    r1 = {r["idx"]: r for r in load("k2bA_r1.json")}
    r2 = {r["idx"]: r for r in load("k2bA_r2.json")}
    same = sum(1 for i in r1 if i in r2 and r1[i]["retrieved_contexts"] == r2[i]["retrieved_contexts"])
    assert same == len(r1) == len(r2), (
        f"组2 两轮上下文并非逐条一致（{same}/{len(r1)}），禁止复用 R1 判定"
    )
    print(f"复用前提通过：组2 两轮检索上下文 {same}/{len(r1)} 逐条一致")

    manifest = {"new_judge": NEW_JUDGE, "old_judge": OLD_JUDGE, "batches": {}}
    for out_name, base, partial, glm, fallback in SPEC:
        rows = load(base)
        rel = new_relevancy(partial)
        old = glm_scores(glm) if glm else {}
        fb = glm_scores(fallback) if fallback else {}
        built, missing_rel, borrowed = [], [], 0
        for r in rows:
            i = int(r["idx"])
            s = {k: None for k in CORE}
            s["answer_relevancy"] = rel.get(i)
            if s["answer_relevancy"] is None:
                missing_rel.append(i)
            for k in ("faithfulness", "context_precision", "context_recall"):
                if i in old and old[i].get(k) is not None:
                    s[k] = old[i][k]
                elif fallback and k in ("context_precision", "context_recall") \
                        and i in fb and fb[i].get(k) is not None:
                    # 只复用不含 response 的两项，faithfulness 保持 None
                    s[k] = fb[i][k]
                    borrowed += 1
            built.append({**{k: r[k] for k in KEEP}, "scores": s})

        path = TMP / f"{out_name}.json"
        json.dump(built, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        manifest["batches"][out_name] = {
            "n": len(built),
            "answer_relevancy": NEW_JUDGE,
            "other_metrics": OLD_JUDGE,
            "borrowed_from_r1": borrowed if fallback else 0,
            "faithfulness_none": sum(1 for b in built if b["scores"]["faithfulness"] is None),
            "missing_relevancy": missing_rel,
        }
        print(f"  {out_name}: {len(built)} 题 | 复用 R1 context 判定 {borrowed} 个 | "
              f"忠实度留空 {manifest['batches'][out_name]['faithfulness_none']} | "
              f"相关性缺失 {len(missing_rel)}")

    json.dump(manifest, open(TMP / "mixed_manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("来源清单 -> .tmp/ragas/mixed_manifest.json")


if __name__ == "__main__":
    main()
