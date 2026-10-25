"""
会话记忆检索：查询重写 + 混合向量检索（仅当前 memory_scope）。
包括用于构建 Agent 工具的记忆检索和用于补充系统提示词的预检索。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from app.chat.milvus_memory import get_chat_memory_milvus, memory_filter_expr
from app.chat.memory_scope import memory_scope_for
from app.kb.multimodal_embedding import get_multimodal_embedding_service
from app.settings import settings
from app.utils.llm_factory import LlmFactory

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """你是查询重写助手。将用户输入改写成适合向量检索「历史对话」的简短查询（1~3 句中文或关键词），要求：
- 消除指代（如「那个」「上次」补全为具体主题）；
- 不要回答问题、不要解释；
- 只输出重写后的查询文本，不要其他内容。

用户输入：
{query}

重写后的检索查询："""


def rewrite_memory_query(raw_query: str, llm_config: dict[str, Any]) -> str:
    """
    重写会话记忆查询, 用于重写用户输入, 使其更适合向量检索历史对话记忆。
    :param raw_query: 原始查询
    :param llm_config: LLM配置
    :return: 重写后的查询
    """
    q = (raw_query or "").strip()
    if not q:
        return ""
    key = (llm_config.get("api_key") or "").strip()
    if not key:
        return q
    try:
        model = LlmFactory.try_chat(llm_config)
        if model is None:
            return q
        out = model.invoke([HumanMessage(content=REWRITE_PROMPT.format(query=q))])
        text = (getattr(out, "content", None) or str(out)).strip()
        return text if text else q
    except Exception:
        logger.exception("rewrite_memory_query failed, using raw query")
        return q


def retrieve_session_memory_hits(
    query: str,
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    llm_config: dict[str, Any],
    top_k: int,
    allowed_turn_keys: list[int] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    混合检索会话记忆；返回 (结果列表, 重写后的检索串)。
    无命中或不可用时结果为空列表，重写串仍可能非空。
    allowed_turn_keys：只检索这些轮次的记忆（当前路径上的已归档轮；空列表表示无可检索轮次）。
    """
    empty: list[dict[str, Any]] = []
    if not (settings.EMBEDDING_API_KEY or "").strip():
        return empty, ""
    q = (query or "").strip()
    if not q:
        return empty, ""
    if allowed_turn_keys is not None and not allowed_turn_keys:
        return empty, ""

    mem_scope = memory_scope_for(user_id, agent_id, session_id)  # 获取会话记忆的隔离键, 只能检索当前会话的记忆
    rewritten = rewrite_memory_query(q, llm_config)  # 重写用户输入, 使其更适合向量检索历史对话记忆。
    search_q = (rewritten or q).strip()
    if not search_q:
        return empty, rewritten

    try:
        embedder = get_multimodal_embedding_service()
        dense = embedder.get_text_embeddings([search_q[:8000]])
        milvus = get_chat_memory_milvus()
        milvus.init_collection()
        flt = memory_filter_expr(mem_scope, turn_keys=allowed_turn_keys)
        hits = milvus.hybrid_retrieve(
            dense[0],
            search_q[:8000],
            top_k=max(1, top_k),
            filter_expr=flt,
        )
        if allowed_turn_keys is not None:
            allowed = {int(k) for k in allowed_turn_keys}
            hits = [h for h in hits if int(h.get("turn_key", -1) or -1) in allowed]
        return hits, rewritten
    except Exception:
        logger.exception("retrieve_session_memory_hits failed")
        return empty, rewritten


def search_session_memory(
    query: str,
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    llm_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    返回 (格式化片段文本, trace 字典)。将检索到的记忆转换为文本格式, 并返回 trace 字典。
    :param query: 查询
    :param user_id: 用户ID
    :param agent_id: 智能体ID
    :param session_id: 会话ID
    :param llm_config: LLM配置
    :return: 格式化片段文本和 trace 字典
    """
    trace: dict[str, Any] = {"tool_used": True, "tool_name": "search_session_memory", "query": query}
    if not (settings.EMBEDDING_API_KEY or "").strip():
        trace["error"] = "未配置 EMBEDDING_API_KEY"
        return ("会话记忆检索不可用：未配置嵌入服务。", trace)

    top_k = max(1, int(getattr(settings, "CHAT_MEMORY_SEARCH_TOP_K", 5) or 5))
    from app.chat.memory_archive import archived_turn_keys_on_path

    allowed_keys = archived_turn_keys_on_path(user_id, agent_id, session_id)
    hits, rewritten = retrieve_session_memory_hits(
        query.strip(),
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        llm_config=llm_config,
        top_k=top_k,
        allowed_turn_keys=allowed_keys,
    )
    trace["rewritten_query"] = rewritten

    trace["hit_count"] = len(hits)
    if not hits:
        return ("当前会话较早的对话中未检索到与问题相关的片段。", trace)

    lines: list[str] = ["[会话历史记忆] 以下为检索到的较早对话片段（按相关度排序）："]
    for i, h in enumerate(hits, 1):
        ti = h.get("turn_index", "")
        tx = (h.get("text") or "").strip()
        lines.append(f"[{i}] 轮次 {ti}：\n{tx}")
    return "\n\n".join(lines), trace


def proactive_session_memory_inject_text(
    user_query: str,
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    llm_config: dict[str, Any],
    path_turn_keys: list[int] | None = None,
) -> str | None:
    """
    用本轮用户输入预检索，返回可拼入 System 的摘录正文；无命中或关闭功能时返回 None。
    这部分记忆检索是预先检索的, 用于在用户输入后立即检索, 用于补充系统提示词。和 Agent 工具检索互补。
    :param user_query: 用户输入
    :param user_id: 用户ID
    :param agent_id: 智能体ID
    :param session_id: 会话ID
    :param llm_config: LLM配置
    :return: 可拼入 System 的摘录正文
    """
    #1. 开关检查（三重保险）
    if not getattr(settings, "CHAT_USE_SESSION_MEMORY", True):
        return None
    if not getattr(settings, "CHAT_MEMORY_PROACTIVE_INJECT", True):
        return None
    if not (settings.EMBEDDING_API_KEY or "").strip():
        return None
    q = (user_query or "").strip()
    if not q:
        return None

    top_k = max(1, int(getattr(settings, "CHAT_MEMORY_PROACTIVE_TOP_K", 3) or 3))
    from app.chat.memory_archive import archived_turn_keys_on_path

    #2. 计算允许检索的轮次范围
    allowed_keys = archived_turn_keys_on_path(user_id, agent_id, session_id, path_turn_keys)
    #3. 混合向量检索（top_k=3）
    hits, _rew = retrieve_session_memory_hits(
        q,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        llm_config=llm_config,
        top_k=top_k,
        allowed_turn_keys=allowed_keys,
    )
    if not hits:
        return None

    #4. 格式化输出
    # 显式加标记：U7 起注入了「长期记忆」和「本会话摘录」两段不同来源，
    # 模型需要分得清哪段是原话、哪段是提炼过的事实；思考区也靠这个标记如实显示。
    lines: list[str] = []
    for i, h in enumerate(hits, 1):
        ti = h.get("turn_index", "")
        tx = (h.get("text") or "").strip()
        lines.append(f"[{i}] 轮次 {ti}：\n{tx}")
    return SESSION_EXCERPT_INJECT_MARK + "\n" + "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 注入来源标记：思考区据此如实显示「注入了什么」，而不是写死一句话
# ---------------------------------------------------------------------------

LONG_TERM_INJECT_MARK = "【长期记忆】"
SESSION_EXCERPT_INJECT_MARK = "【本会话较早摘录】"


def inject_sources(text: str) -> list[str]:
    """判断一次注入里含哪几段来源，供思考区显示。

    为什么需要：U7 把两段不同来源拼成一次注入（会话级原话摘录 + 用户级提炼事实），
    如果思考区统一写「会话记忆预注入 / 已附加较早轮次摘录」，在**全新会话**里就会
    显示一件根本不存在的事——新会话没有任何被归档的轮次，用户会以为系统在编。
    标签必须由实际内容决定，不能由调用点猜。

    :return: 例 ["长期记忆"] / ["本会话较早摘录"] / 两者都有时两段
    """
    t = text or ""
    srcs: list[str] = []
    if LONG_TERM_INJECT_MARK in t:
        srcs.append("长期记忆")
    if SESSION_EXCERPT_INJECT_MARK in t:
        srcs.append("本会话较早摘录")
    return srcs


# ---------------------------------------------------------------------------
# U7：跨会话长期记忆检索
# ---------------------------------------------------------------------------


def _parse_memitem_id(chunk_id: str) -> int | None:
    """从 `memitem_{pg_id}` 里解出 PG 主键。"""
    cid = (chunk_id or "").strip()
    if not cid.startswith("memitem_"):
        return None
    try:
        return int(cid[len("memitem_") :])
    except (TypeError, ValueError):
        return None


def retrieve_user_memory_hits(
    query: str,
    *,
    user_id: int,
    agent_id: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """用户级（跨会话）记忆的语义检索。失败返回空列表，绝不抛。

    **Milvus 为主、PG 兜底**：双写允许落后（`embedding_synced=false` 的条目还没进
    Milvus），如果只查 Milvus，用户会遇到"明明记过却查不到"。所以命中为空时
    回退到 PG 最近更新的几条——语义上不如向量检索准，但保证"记忆不丢"。
    """
    from app.chat.memory_curation import memory_user_scope, user_scope_enabled
    from app.chat.memory_store import list_active, list_by_ids

    if not user_scope_enabled():
        return []
    q = (query or "").strip()
    if not q:
        return []

    hits: list[dict[str, Any]] = []
    if (settings.EMBEDDING_API_KEY or "").strip():
        try:
            embedder = get_multimodal_embedding_service()
            dense = embedder.get_text_embeddings([q[:8000]])
            milvus = get_chat_memory_milvus()
            milvus.init_collection()
            scope = memory_user_scope(user_id, agent_id)
            raw = milvus.hybrid_retrieve(
                dense[0],
                q[:8000],
                top_k=max(1, top_k),
                filter_expr=f'memory_scope == "{scope}"',
            )
            ids = [i for i in (_parse_memitem_id(h.get("chunk_id", "")) for h in raw) if i]
            meta = list_by_ids(user_id, agent_id, ids)
            for h in raw:
                mid = _parse_memitem_id(h.get("chunk_id", ""))
                if not mid:
                    continue
                row = meta.get(mid)
                # 只认 active：Milvus 的删除是异步生效的，可能短暂残留已失效条目
                if not row or row.get("status") != "active":
                    continue
                hits.append(
                    {
                        "id": mid,
                        "text": row["text"],
                        "mem_key": row["mem_key"],
                        "kind": row["kind"],
                        "score": h.get("score", 0.0),
                        "source": "milvus",
                    }
                )
        except Exception:
            logger.exception("retrieve_user_memory_hits: milvus 失败，回退 PG")

    if not hits:
        for row in list_active(user_id, agent_id, limit=max(1, top_k)):
            hits.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "mem_key": row["mem_key"],
                    "kind": row["kind"],
                    "score": None,
                    "source": "pg_recent",
                }
            )
    return hits


def format_user_memory_hits(hits: list[dict[str, Any]]) -> str:
    """把命中条目格式化成给模型看的文本。"""
    if not hits:
        return ""
    lines = [LONG_TERM_INJECT_MARK + "以下是关于这位用户/这个项目、在以往会话中确认过的事实："]
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}]（{h.get('kind')}）{h.get('text')}")
    return "\n".join(lines)


def search_long_term_memory(
    query: str,
    *,
    user_id: int,
    agent_id: int,
) -> tuple[str, dict[str, Any]]:
    """工具入口：返回 (格式化文本, trace)。"""
    trace: dict[str, Any] = {
        "tool_used": True,
        "tool_name": "search_long_term_memory",
        "query": query,
    }
    top_k = max(1, int(getattr(settings, "CHAT_MEMORY_USER_SCOPE_SEARCH_TOP_K", 5) or 5))
    hits = retrieve_user_memory_hits(query, user_id=user_id, agent_id=agent_id, top_k=top_k)
    trace["hit_count"] = len(hits)
    trace["source"] = hits[0]["source"] if hits else None
    if not hits:
        return ("以往会话中没有相关的长期记忆。", trace)
    return (format_user_memory_hits(hits), trace)
