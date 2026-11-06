# -*- coding: utf-8 -*-
"""采集 RAGAS 输入：对 Chiron-AI 的 RAG 子图取「完整上下文」，再接地生成回答。

为什么要自己写这一层（而不是直接用 scripts/eval_retrieval.py）：
  · eval_retrieval.py 的 --json 里 text 被截到 `text_head`（120 字），喂给 RAGAS 会让
    context_precision / faithfulness 全部失真（交接文档 7.1 坑 1）。
  · 它没有参考答案、没有数据集概念，是单题探针不是评估器。

本脚本做三件事（一篇一个问题，串行）：
  ① run_rag_graph(question, kb_scope, llm_config)  → docs（含完整 text）
  ② 把 docs 用**项目自己的** format_knowledge_retrieval_tool_output 拼成"模型实际看到的那段工具输出"
  ③ 用智能体配置的模型（解密 api_key_ciphertext）接地回答 → response

★ 与真实对话链路的差异（写进报告，别当成缺陷）
  真实链路是 Agent + 工具选择 + 记忆 + 可能联网；本脚本固定「先检索一次、再基于该结果作答」。
  对 RAGAS（faithfulness / precision / recall）来说这正是该测的对象：
  上下文与回答来自同一次检索，不存在"上下文和回答对不上"的错配。

★ 口径开关 --rag-llm
  off（默认）：不给 RAG 子图传 llm_config → 跳过逐块打分与改写，检索确定性强，基线可复现。
  on        ：传 llm_config → 走完整的「检索→逐块打分→改写→再检索」自纠正链路（花钱、且不确定）。

用法：
  ./.venv/Scripts/python.exe scripts/ragas/collect_chiron.py \
      --questions=.tmp/ragas/questions_smoke.json --agent-id=2 --out=.tmp/ragas/collected_smoke.json

  # 只取上下文不生成回答（0 成本调参用）
  ... --dump-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SYS_KB_RULE_PLAIN = (
    "\n\n下面是知识库检索结果。请直接用连贯的段落回答问题："
    "不要使用列表、编号、表格或代码块，不要在文中标注任何来源编号，"
    "用 2~4 句话说清结论与关键依据即可。检索结果不足以回答时如实说明。"
)

_SYS_KB_RULE = (
    "\n\n你收到的是「知识库检索结果」。回答纪律：\n"
    "1) 只依据检索结果作答，不得使用检索结果以外的知识硬编结论；\n"
    "2) 凡引用检索内容，必须以 [来源N] 标注（N 与检索结果编号一致）；\n"
    "3) 检索结果不足以回答时，如实说明知识库中没有相关资料，不要猜测。"
)


def _parse(argv):
    p = argparse.ArgumentParser(description="采集 Chiron-AI RAG 的 RAGAS 输入")
    p.add_argument("--questions", required=True, help="问题+参考答案 JSON")
    p.add_argument("--agent-id", type=int, required=True, help="用于取模型配置与 kb_scope")
    p.add_argument("--out", required=True, help="输出 collected.json")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    p.add_argument("--rag-llm", choices=["on", "off"], default="off", help="是否给 RAG 子图传 llm_config")
    p.add_argument("--dump-only", action="store_true", help="只取上下文，不生成回答")
    p.add_argument("--vision", action="store_true",
                   help="把图片块的像素一并喂给模型（真实多模态链路）。默认只拼图片链接，等于纯文本通道")
    p.add_argument("--max-images", type=int, default=2, help="--vision 时最多送几张图")
    p.add_argument("--style", choices=["default", "plain"], default="default",
                   help="回答风格。plain = 段落式作答、不列清单、不加引用标记（用于判定指标口味）")
    p.add_argument("--sleep", type=float, default=0.0, help="题间间隔秒数")
    return p.parse_args(argv)


def _pg_dsn() -> str:
    from app.settings.config import settings

    url = settings.DATABASE_URL
    for prefix in ("postgresql+psycopg2://", "postgresql+psycopg://", "postgresql+asyncpg://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix):]
    return url


def _load_agent(agent_id: int) -> dict:
    """取智能体模型配置并解密 api_key（与 agent_service._llm_config_from_ua 同口径）。"""
    import psycopg

    from app.utils.api_key_crypto import decrypt_api_key_safe

    with psycopg.connect(_pg_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT user_id, name, model_name, base_url, api_key_ciphertext, system_prompt '
                'FROM user_agent WHERE id = %s',
                (agent_id,),
            )
            row = cur.fetchone()
    if not row:
        raise SystemExit(f"智能体 {agent_id} 不存在")
    user_id, name, model_name, base_url, cipher, sys_prompt = row
    api_key = (decrypt_api_key_safe(cipher) or "").strip()
    if not api_key and not base_url:
        raise SystemExit(f"智能体 {agent_id} 没有可用的模型配置（api_key 解密失败且无 base_url）")
    return {
        "user_id": user_id,
        "name": name,
        "model_name": model_name,
        "base_url": (base_url or "").strip(),
        "api_key": api_key,
        "system_prompt": (sys_prompt or "").strip(),
    }


def _load_questions(path: str, limit: int) -> list[dict]:
    raw = json.load(open(path, encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("questions", [])
    out = []
    for i, r in enumerate(rows, 1):
        q = (r.get("q") or r.get("question") or "").strip()
        ref = (r.get("reference") or "").strip()
        if not q:
            continue
        if not ref:
            print(f"  ⚠️ 第 {i} 题缺参考答案，RAGAS 会跳过 precision/recall：{q[:20]}", flush=True)
        out.append({"idx": r.get("idx", i), "q": q, "reference": ref, "expect": r.get("expect", "answer")})
    return out[:limit] if limit else out


def _format_ctx(docs: list[dict]) -> str:
    """模型实际看到的那段工具输出（走项目自己的格式化器；失败则退化为纯文本拼接）。"""
    try:
        from app.kb.kb_tool_formatting import format_knowledge_retrieval_tool_output

        text, _imgs, _srcs = format_knowledge_retrieval_tool_output(docs)
        return text
    except Exception as e:  # 图片签名等依赖 Web 上下文，脚本里可能不可用
        print(f"  ⚠️ 工具输出格式化失败（退化为纯文本拼接）：{e}", flush=True)
        blocks = []
        for i, d in enumerate(docs, 1):
            text = (d.get("text") or "").strip()
            if text:
                blocks.append(f"[{i}] {d.get('filename')} (Page {d.get('page_number')})\n{text}")
        return "\n\n".join(blocks)


def _image_parts(docs: list[dict], max_images: int, max_bytes: int = 4 * 1024 * 1024) -> list[dict]:
    """把检索命中的图片块的**像素**读出来，转成 OpenAI 兼容的 image_url part（data URI）。

    为什么需要：文本块里没有的内容（截图里的 SQL、表格、公式）只在图片像素里。
    只拼 markdown 链接等于让模型看不见图，测出来的低分不是产品能力。
    """
    import base64

    from app.core import object_storage as obs
    from app.settings.config import settings

    parts: list[dict] = []
    for d in docs:
        if len(parts) >= max_images:
            break
        if (d.get("content_type") or "") != "image":
            continue
        meta = d.get("image_metadata") or {}
        rel = (meta.get("stored_relpath") or "").strip()
        if not rel:
            continue
        try:
            raw = obs.read_bytes(obs.join_key(settings.USER_AGENT_KB_IMAGES_ROOT, rel))
        except Exception as e:
            print(f"  ⚠️ 读图失败 {rel}: {e}", flush=True)
            continue
        if len(raw) > max_bytes:
            print(f"  ⚠️ 图片超限跳过 {rel} ({len(raw)} bytes)", flush=True)
            continue
        fmt = (meta.get("format") or "png").lower().lstrip(".")
        uri = f"data:image/{fmt};base64," + base64.b64encode(raw).decode()
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    return parts


def _generate(client, model: str, system_prompt: str, question: str, ctx: str,
              image_parts: list[dict] | None = None, style: str = "default") -> tuple[str, dict]:
    rule = _SYS_KB_RULE_PLAIN if style == "plain" else _SYS_KB_RULE
    sys_msg = (system_prompt or "你是一个乐于助人的助手。") + rule
    user_text = (
        f"用户问题：{question}\n\n"
        f"【知识库检索结果】\n{ctx}\n\n"
        f"请基于以上检索结果回答问题。"
    )
    if image_parts:
        user_content: object = [{"type": "text", "text": user_text}, *image_parts]
    else:
        user_content = user_text
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_content}],
        temperature=0.0,
    )
    usage = getattr(resp, "usage", None)
    meta = {
        "gen_sec": round(time.time() - t0, 1),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "n_images_sent": len(image_parts or []),
    }
    return (resp.choices[0].message.content or "").strip(), meta


def main(argv=None) -> int:
    args = _parse(argv)

    from app.kb.kb_scope import kb_scope_for
    from app.kb.milvus_client import MilvusManager
    from app.kb.rag_pipeline import run_rag_graph

    agent = _load_agent(args.agent_id)
    questions = _load_questions(args.questions, args.limit)
    if not questions:
        raise SystemExit("没有可跑的问题")

    kb_scope = kb_scope_for(agent["user_id"], args.agent_id)
    llm_config = (
        {"api_key": agent["api_key"], "base_url": agent["base_url"], "model_name": agent["model_name"]}
        if args.rag_llm == "on"
        else {}
    )

    MilvusManager().init_collection()

    client = None
    if not args.dump_only:
        from openai import OpenAI

        client = OpenAI(api_key=agent["api_key"], base_url=agent["base_url"] or None, timeout=300)

    print(
        f"kb_scope={kb_scope}  题数={len(questions)}  rag_llm={args.rag_llm}  "
        f"回答模型={agent['model_name']}  dump_only={args.dump_only}",
        flush=True,
    )

    rows = []
    for item in questions:
        q = item["q"]
        t0 = time.time()
        state = run_rag_graph(q, kb_scope, llm_config)
        rag_sec = round(time.time() - t0, 1)

        docs = state.get("docs") or []
        # ★ 只取有正文的块：图片块 text 为空，喂给 RAGAS 会拉低 precision 且不携带语义
        ctx_pairs = [(d, (d.get("text") or "").strip()) for d in docs]
        contexts = [t for _d, t in ctx_pairs if t]
        no_answer = bool(state.get("no_answer"))
        trace = state.get("rag_trace") or {}

        if args.dump_only:
            response, gen_meta = "", {}
        elif no_answer or not contexts:
            # 与真实链路一致：无相关资料时给拒答指令，而不是让模型在空上下文上硬编
            refuse = (
                "KNOWLEDGE_BASE_NO_RELEVANT_INFO: 本知识库经两次检索与逐块相关性评估，"
                "未找到与用户问题相关的资料。请如实告知用户：知识库中没有相关资料，"
                "可建议用户上传补充相关资料后重试；不要引用任何检索片段，也不要编造任何知识库结论。"
            )
            response, gen_meta = _generate(
                client, agent["model_name"], agent["system_prompt"], q, refuse, style=args.style
            )
        else:
            # ★ --vision：把命中的图片像素一起送进去，还原真实多模态链路；
            #   不加这个开关就只拼图片 markdown 链接，等于只测纯文本通道。
            imgs = _image_parts(docs, args.max_images) if args.vision else []
            response, gen_meta = _generate(
                client, agent["model_name"], agent["system_prompt"], q, _format_ctx(docs), imgs,
                style=args.style,
            )

        row = {
            "idx": item["idx"],
            "user_input": q,
            "response": response,
            "reference": item["reference"],
            "retrieved_contexts": contexts,
            "source_meta": [
                {
                    "filename": d.get("filename"),
                    "page_number": d.get("page_number"),
                    "content_type": d.get("content_type") or "text",
                    "score": d.get("score"),
                }
                for d, _t in ctx_pairs
            ],
            "rag": {
                "kb_scope": kb_scope,
                "no_answer": no_answer,
                "n_docs": len(docs),
                "n_ctx": len(contexts),
                "n_empty_text": len(docs) - len(contexts),
                "expect": item["expect"],
                "rag_sec": rag_sec,
                **gen_meta,
                "trace": {
                    "grade_score": trace.get("grade_score"),
                    "grade_route": trace.get("grade_route"),
                    "rewrite_needed": trace.get("rewrite_needed"),
                    "rewrite_strategy": trace.get("rewrite_strategy"),
                    "rewrite_query": trace.get("rewrite_query"),
                    "second_grade": trace.get("second_grade"),
                    "auto_merge_applied": trace.get("auto_merge_applied"),
                },
            },
        }
        rows.append(row)
        print(
            f"[{row['idx']:2d}] ctx={len(contexts)}(空{row['rag']['n_empty_text']}) "
            f"图={gen_meta.get('n_images_sent', 0)} "
            f"no_answer={no_answer} rag={rag_sec}s gen={gen_meta.get('gen_sec', '-')}s "
            f"resp={len(response)}字  {q[:28]}",
            flush=True,
        )
        if args.sleep:
            time.sleep(args.sleep)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    pt = sum(r["rag"].get("prompt_tokens") or 0 for r in rows)
    ct = sum(r["rag"].get("completion_tokens") or 0 for r in rows)
    print(f"\n-> {out}")
    print(f"token 合计：in={pt} out={ct}；拒答 {sum(1 for r in rows if r['rag']['no_answer'])}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
