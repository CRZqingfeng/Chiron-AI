"""独立检索评测脚本：不建 FastAPI 应用，不连 Redis，只跑 RAG 子图。

它存在的意义只有一个——证明「检索真的解耦了」。
U8 之前，任何 `import app.kb.rag_pipeline` 都会因为 `app/__init__.py` 里的
`app = create_app()` 把整个 Web 应用拉起来（37 个 app.chat.* 模块 + FastAPI +
Tortoise），想离线评测检索只能连上全套基础设施。

用法（在仓库根目录）：
    # 纯检索：不给 api_key，评分/改写环节自动跳过，只验召回
    python scripts/eval_retrieval.py --user-id 1 --agent-id 2 --question "退货政策怎么规定"

    # 带模型：相关性打分与查询改写都会真实调用
    python scripts/eval_retrieval.py --user-id 1 --agent-id 2 --question "..." \\
        --api-key sk-xxx --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --model qwen-plus

    # 只导出 JSON，便于脚本对比
    python scripts/eval_retrieval.py ... --json

退出码：0 成功；2 参数错误；3 Milvus 不可用；4 检索抛异常。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _environment_report() -> dict:
    """自检：确认没有把 Web 应用和对话域拖起来。这是解耦的硬证据。"""
    chat_modules = sorted(m for m in sys.modules if m.startswith("app.chat"))
    return {
        "app_chat_modules": len(chat_modules),
        "app_main_imported": "app.main" in sys.modules,
        "fastapi_app_built": _has_fastapi_app(),
    }


def _has_fastapi_app() -> bool:
    """有没有实例化的 FastAPI 对象（模块被 import 不算，要真建出 app）。"""
    import gc

    try:
        from fastapi import FastAPI
    except ImportError:
        return False
    return any(isinstance(obj, FastAPI) for obj in gc.get_objects())


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="独立检索评测（不启动 Web 服务）")
    p.add_argument("--user-id", type=int, required=True, help="知识库属主用户 ID")
    p.add_argument("--agent-id", type=int, required=True, help="智能体 ID")
    p.add_argument("--question", required=True, help="检索问题")
    p.add_argument("--top-k", type=int, default=5, help="返回片段数（默认 5）")
    p.add_argument("--api-key", default="", help="可选：给了才做相关性打分与查询改写")
    p.add_argument("--base-url", default="", help="LLM base_url（OpenAI 兼容）")
    p.add_argument("--model", default="", help="模型名，默认取 settings 或 gpt-4")
    p.add_argument("--json", action="store_true", help="只输出 JSON 结果")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # 延迟到参数校验之后才 import：让 --help 不触发任何连接
    from app.kb.kb_scope import kb_scope_for
    from app.kb.milvus_client import MilvusManager
    from app.kb.rag_pipeline import run_rag_graph

    env = _environment_report()

    try:
        MilvusManager().init_collection()
    except Exception as e:
        print(f"Milvus 不可用：{e}", file=sys.stderr)
        return 3

    llm_config: dict = {}
    if args.api_key.strip():
        llm_config = {
            "api_key": args.api_key.strip(),
            "base_url": args.base_url.strip(),
            "model_name": args.model.strip(),
        }

    kb_scope = kb_scope_for(args.user_id, args.agent_id)
    try:
        state = run_rag_graph(args.question, kb_scope, llm_config)
    except Exception as e:
        print(f"检索失败：{e}", file=sys.stderr)
        return 4

    docs = state.get("docs") or []
    result = {
        "kb_scope": kb_scope,
        "question": args.question,
        "with_llm": bool(llm_config),
        "docs_count": len(docs),
        "docs": [
            {
                "filename": d.get("filename"),
                "page_number": d.get("page_number"),
                "content_type": d.get("content_type") or "text",
                "score": d.get("score"),
                "text_head": (d.get("text") or "")[:120],
            }
            for d in docs[: args.top_k]
        ],
        "rag_trace": state.get("rag_trace") or {},
        "no_answer": bool(state.get("no_answer")),
        "environment": env,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"知识库范围：{kb_scope}")
    print(f"问题：{args.question}")
    print(f"命中片段：{len(docs)} 条" + ("" if llm_config else "（未配模型：跳过打分与改写）"))
    for i, d in enumerate(result["docs"], 1):
        score = d.get("score")
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        print(f"  [{i}] {d['filename']} p{d['page_number']}  score={score_s}")
        print(f"      {d['text_head']}")
    trace = result["rag_trace"]
    if trace:
        print("rag_trace：")
        for k, v in trace.items():
            if k == "per_chunk_grades":
                print(f"  {k}: {len(v or [])} 条评分")
            else:
                # 检索中间态里含整段原文，终端打印只留摘要，完整内容走 --json
                text = str(v)
                if len(text) > 160:
                    text = f"{text[:160]}…（{len(text)} 字符，--json 看全量）"
                print(f"  {k}: {text}")
    print("环境自检：")
    print(f"  app.chat.* 模块数：{env['app_chat_modules']}（要求 0）")
    print(f"  app.main 被导入：{env['app_main_imported']}（要求 False）")
    print(f"  FastAPI 实例已创建：{env['fastapi_app_built']}（要求 False）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
