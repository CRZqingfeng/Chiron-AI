"""智能体系统提示、本轮上下文与调试回调。"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.core.message_codec import msg_content_to_str
from app.models.user_agent import UserAgent
from app.settings import settings
from app.utils.content_guard import guard_untrusted_content


def _format_one_message_for_debug(msg: BaseMessage) -> str:
    """
    格式化一条消息，用于调试
    :param msg: 消息
    :return: 字符串
    """
    if isinstance(msg, SystemMessage):
        return f"[System]\n{msg_content_to_str(msg.content)}"
    if isinstance(msg, HumanMessage):
        return f"[Human]\n{msg_content_to_str(msg.content)}"
    if isinstance(msg, AIMessage):
        blocks = ["[AI]"]
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            blocks.append(f"tool_calls: {json.dumps(tool_calls, ensure_ascii=False)}")
        body = msg_content_to_str(getattr(msg, "content", ""))
        if body:
            blocks.append(body)
        return "\n".join(blocks)
    if isinstance(msg, ToolMessage):
        name = getattr(msg, "name", "") or ""
        return f"[Tool:{name}]\n{msg_content_to_str(msg.content)}"
    return f"[{type(msg).__name__}]\n{msg_content_to_str(getattr(msg, 'content', ''))}"


class _KbPromptDebugCallback(BaseCallbackHandler):
    """在每次 LLM 调用前打印完整输入消息列表（受 DEBUG_AGENT_KB_PROMPT 控制）。"""

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        from app.settings import settings

        if not getattr(settings, "DEBUG_AGENT_KB_PROMPT", False):
            return
        for batch in messages:
            if not batch:
                continue
            sep = "=" * 72
            lines: list[str] = []
            for i, m in enumerate(batch):
                lines.append(f"--- message[{i}] ---")
                lines.append(_format_one_message_for_debug(m))
            out = "\n".join(lines)
            print(
                f"\n{sep}\n[智能体 LLM] 本轮发给模型的完整消息列表:\n{sep}\n{out}\n{sep}\n",
                flush=True,
            )


def _recursion_limit() -> int:
    """单轮最大推理步数（模型调用与工具往返都算一步）。

    为什么不再是硬编码 30：2026-10-24 线上实录，用户让模型「写一篇 MySQL 文档、
    写进工作区、再配套练习文件」，检索 + 多次读写文件每样都要一轮，30 步直接打穿，
    用户看到「已达到单轮最大推理步数」且正文为空。规划（write_todos）还会多占几步。
    下限兜到 10：配错成 0/负数会让每轮立刻失败，比放宽更糟。
    """
    raw = getattr(settings, "CHAT_RECURSION_LIMIT", 60)
    # 注意别写成 `raw or 60`：那会把显式的 0 当成"没配"，用户设 0 却按 60 走，
    # 排查时会怀疑人生。只有 None 才是"没配"。
    if raw is None:
        return 60
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 60
    return max(10, value)


def _agent_invoke_config(thread_id: str | None = None) -> dict[str, Any]:
    """调用配置。

    :param thread_id: 传给 checkpointer 的会话线程标识。只有在启用审批中断时才有值——
        没有它，`interrupt()` 无处落盘，也就无法 resume。用 job_id 而不是 session_id：
        每轮对话都是「全量历史一次性喂入」，用 session_id 会让 checkpoint 累积重复历史。
    """
    cfg: dict[str, Any] = {
        "recursion_limit": _recursion_limit(),
        "callbacks": [_KbPromptDebugCallback()],
    }
    if thread_id:
        cfg["configurable"] = {"thread_id": thread_id}
    return cfg


# 三份工具纪律已迁到各自 provider（U6b）：工具定义与「该怎么用」住在同一个文件，
# 改一处不会只改一半。这里保留同名的模块级别名，避免外部引用断裂。
from app.chat.tool_providers.kb import (  # noqa: E402
    _KB_ANSWER_DISCIPLINE,
    _KB_IMAGE_DISCIPLINE,
)
from app.chat.tool_providers.web import _WEB_SEARCH_DISCIPLINE  # noqa: E402


def _format_kb_scope_for_turn(document_filter: list[str] | None) -> str:
    """本回合选档范围（写入最后一条 Human，不进 system）。"""
    if document_filter is None:
        return (
            "【本回合知识库检索范围】未限定在单批 file_key（将按当前智能体整库检索；无文档时无结果）。"
            "调用 search_knowledge_base 时只需传入 query。"
        )
    lines = "\n".join(f"- `{x}`" for x in document_filter)
    return (
        "【本回合知识库检索范围】已在回复前由系统根据用户问题将检索**限定在以下文档**（file_key）。\n"
        f"{lines}\n"
        "调用 search_knowledge_base 时只传 query，不要尝试指定或更换 file_key / 范围。"
    )


def _format_turn_context_block(
    *,
    use_knowledge_retrieval: bool,
    use_web_search: bool,
    document_filter: list[str] | None,
    session_attachment_hint: str,
    memory_inject: str | None,
    mcp_approval_note: str | None,
    image_caption: str | None = None,
) -> str:
    """拼装仅本回合有效、将并入最后一条 Human 的上下文（不落库）。"""
    parts: list[str] = ["【本轮上下文（仅本回合有效）】"]
    # 1. 工具调用范围（三选一）
    if use_web_search:
        parts.append(
            "本轮允许调用 web_search / fetch_url / web_image_search；"
            "不要调用 search_knowledge_base / search_knowledge_by_image。"
        )
    elif use_knowledge_retrieval:
        parts.append(
            "本轮允许调用 search_knowledge_base / search_knowledge_by_image；"
            "不要调用 web_search / fetch_url / web_image_search。"
        )
    else:
        parts.append(
            "本轮不要调用 web_search、fetch_url、web_image_search、"
            "search_knowledge_base、search_knowledge_by_image。"
        )
    # 2. 知识库选档范围（有 document_filter 时才写）
    if use_knowledge_retrieval:
        parts.append(_format_kb_scope_for_turn(document_filter))
    # 3. 附件提示（有就写）
    hint = (session_attachment_hint or "").strip()
    if hint:
        parts.append(hint)
    # 4. 图片描述（两阶段读图成功时）
    caption = (image_caption or "").strip()
    if caption:
        max_chars = max(200, int(getattr(settings, "CHAT_VISION_CAPTION_MAX_CHARS", 1200) or 1200))
        if len(caption) > max_chars:
            caption = caption[:max_chars] + "…（已截断）"
        parts.append(
            "【本回合图片内容理解（由视觉模型预读，仅供检索与作答参考；"
            "若与用户问题矛盾，以用户补充说明为准）】\n\n"
            + guard_untrusted_content(caption, max_chars=max_chars)
        )
        if use_web_search:
            parts.append(
                "若需搜图，web_image_search 的 query 须用上文专名（可加立绘/Q版/手办等风格词），"
                "不要用「这张图」「这个人物」「类似图片」。"
            )
    # 5. 会话记忆（有命中时）
    if (memory_inject or "").strip():
        parts.append(
            "【本回合根据用户最新输入自动检索的较早会话摘录（仅供参考；"
            "若仍不足可再调用 search_session_memory 工具）】\n\n"
            + memory_inject.strip()
        )
    # 6. MCP 审批说明
    if (mcp_approval_note or "").strip():
        parts.append(mcp_approval_note.strip())
    return "\n\n".join(parts)


def _append_turn_context_message(messages: list, block: str) -> list:
    """本轮上下文追加进最后一条 Human（不落库、不新增第二条 user）。无 Human 则不插入。"""
    text = (block or "").strip()
    if not text:
        return list(messages)
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if not isinstance(out[i], HumanMessage):
            continue
        content = out[i].content
        if isinstance(content, str):
            out[i] = HumanMessage(content=content + "\n\n" + text)
        elif isinstance(content, list):
            out[i] = HumanMessage(content=list(content) + [{"type": "text", "text": text}])
        else:
            out[i] = HumanMessage(content=str(content) + "\n\n" + text)
        return out
    return out


def _compose_system_prompt(
    ua: UserAgent,
    *,
    use_knowledge_retrieval: bool = False,
    use_web_search: bool = False,
    fragments: list[str] | None = None,
) -> str:
    """人设 + 工具纪律（agent 按请求构建）。

    :param fragments: 由 tool provider 产出的纪律片段，按 provider 顺序传入。
        传 None 时走旧拼装逻辑（`CHAT_TOOL_REGISTRY_ENABLED=false` 的回退路径）。
    """
    parts: list[str] = [(ua.system_prompt or "").strip() or "You are a helpful assistant."]
    if fragments is not None:
        parts.extend(f for f in fragments if (f or "").strip())
    else:
        # 旧路径：顺序硬编码为「联网 → KB 图片 → KB 作答」。
        # 保留一个版本用于回退，下个版本删除。
        if use_web_search:
            parts.append(_WEB_SEARCH_DISCIPLINE)
        if use_knowledge_retrieval:
            parts.append(_KB_IMAGE_DISCIPLINE)
            parts.append(_KB_ANSWER_DISCIPLINE)
    return "\n\n".join(parts)

