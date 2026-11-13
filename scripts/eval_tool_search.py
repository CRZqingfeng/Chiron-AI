"""工具检索评测：打印每条查询的排序结果与命中率。

调参时看这个，不是看单测。单测只断言"不低于某个线"，看不出是哪几条在退。

用法：
    python scripts/eval_tool_search.py          # 摘要 + 失败明细
    python scripts/eval_tool_search.py -v       # 每条都打印
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chat.tool_search import ToolIndex, build_docs  # noqa: E402
from app.chat.tool_gateway import _server_of  # noqa: E402
from tests.chat.fixtures.tool_catalog import QUERIES, build_tools  # noqa: E402

TOP_K = 5


def main(verbose: bool = False) -> int:
    tools = build_tools()
    docs = build_docs(tools, server_of=_server_of)
    index = ToolIndex(docs)
    print(f"语料：{len(docs)} 个工具，{len(QUERIES)} 条查询，评测 top-{TOP_K}\n")

    hits = 0
    top1 = 0
    fails: list[tuple[str, set[str], list[str]]] = []
    top1_misses: list[tuple[str, set[str], str]] = []

    for query, expected, note in QUERIES:
        ranked = [d.name for _, d in index.rank(query, TOP_K)]
        ok = bool(set(ranked) & expected)
        hits += ok
        if ranked and ranked[0] in expected:
            top1 += 1
        elif ranked:
            top1_misses.append((query, expected, ranked[0]))
        if not ok:
            fails.append((query, expected, ranked))
        if verbose:
            mark = "✅" if ok else "❌"
            print(f"{mark} {query}")
            print(f"   期望：{sorted(expected)}")
            print(f"   实得：{ranked}")
            if note:
                print(f"   备注：{note}")
            print()

    rate = hits / len(QUERIES) * 100
    top1_rate = top1 / len(QUERIES) * 100
    print(f"top-{TOP_K} 命中率：{hits}/{len(QUERIES)} = {rate:.0f}%")
    print(f"top-1 准确率：{top1}/{len(QUERIES)} = {top1_rate:.0f}%")

    if top1_misses:
        print("\ntop-1 排错（top-5 已兜住，模型可自行挑选）：")
        for query, expected, got in top1_misses:
            print(f"  · {query}\n      期望之一 {sorted(expected)}\n      实得第一 {got}")

    if fails:
        print("\n未命中明细：")
        for query, expected, ranked in fails:
            print(f"  ❌ {query}")
            print(f"     期望 {sorted(expected)}")
            print(f"     实得 {ranked}")

    # 排序质量：破坏性工具不该在"纯查找"查询里挤进前面
    probe = "找 issue 相关的工具"
    ranked = [d.name for _, d in index.rank(probe, TOP_K)]
    pos = ranked.index("sentry__delete_issue") if "sentry__delete_issue" in ranked else -1
    print(f"\n[排序质量] 「{probe}」 → {ranked[:3]}")
    print(f"  delete_issue 位置：{pos if pos >= 0 else '未进 top-' + str(TOP_K)}")
    if pos == 0:
        print("  ⚠️ 破坏性工具排第一，说明降权没生效")

    return 0 if rate >= 80 else 1


if __name__ == "__main__":
    raise SystemExit(main("-v" in sys.argv or "--verbose" in sys.argv))
