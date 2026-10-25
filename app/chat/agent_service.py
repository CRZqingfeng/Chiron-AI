"""
LangChain Agent 对话（同步 invoke + 异步 SSE 流式）。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, AsyncIterator

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)

from app.chat.attachment_service import build_storable_human_content, format_attachment_hint
from app.chat.attachment_tools import make_session_attachment_tools
from app.chat.llm_errors import classify_llm_error
from app.chat.memory_archive import schedule_archive_session_memory
from app.chat.memory_tool import make_search_session_memory_tool
from app.core.memory_turns import apply_sliding_window_turns
from app.core.message_codec import msg_content_to_str
from app.chat.vision_expand import expand_messages_for_model
from app.chat.storage import storage
from app.core.turn_runtime import (
    get_last_rag_context,
    get_pending_mcp_confirmations,
    reset_tool_call_guards,
    set_approved_mcp_pending_id,
    set_rag_step_queue,
    set_turn_tool_policy,
)
from app.chat.web_search_tool import (
    make_fetch_url_tool,
    make_web_image_search_tool,
    make_web_search_tool,
)
from app.kb.image_search_tool import make_search_knowledge_by_image_tool
from app.kb.kb_scope import kb_scope_for
from app.kb.search_tool import make_search_knowledge_tool
from app.log import logger
from app.mcp_client.service import load_agent_mcp_tools
from app.models.user_agent import UserAgent
from app.settings import settings
from app.utils.api_key_crypto import decrypt_api_key_safe
from app.utils.llm_factory import LlmFactory


from app.chat.agent_prompt import (
    _WEB_SEARCH_DISCIPLINE,  # noqa: F401  （tests 经由本模块引用此常量）
    _agent_invoke_config,
    _append_turn_context_message,
    _compose_system_prompt,
    _format_turn_context_block,
)
from app.chat.agent_vision import (
    _human_content_has_image_ref,
    _should_run_vision_caption,
    _try_strip_images_middleware,
    _wrap_model_strip_images_after_tools,
)

def _mcp_tools_allowed_for(ua: UserAgent, user_id: int) -> bool:
    """仅智能体属主会话允许加载其 MCP 工具。

    MCP 工具携带属主凭据（如预置的敏感请求头）且由模型自主调用、无用户确认；
    共享（非属主）会话加载后可被对话内容或注入文本驱动，以属主身份执行操作，故默认禁止。
    可通过 SHARE_CHAT_ALLOW_OWNER_MCP_TOOLS=true 恢复旧行为（不推荐）。
    """
    if int(getattr(ua, "user_id", 0) or 0) == int(user_id):
        return True
    return bool(getattr(settings, "SHARE_CHAT_ALLOW_OWNER_MCP_TOOLS", False))


def _llm_config_from_ua(ua: UserAgent) -> dict[str, Any]:
    """
    从智能体配置中获取 LLM 配置
    :param ua: 智能体
    :return: LLM 配置
    """
    plain = decrypt_api_key_safe(ua.api_key_ciphertext)
    base_url = (ua.base_url or "").strip() or None
    if base_url:
        from app.utils.ssrf import assert_public_http_url

        assert_public_http_url(base_url)
    return {
        "api_key": (plain or "").strip(),
        "base_url": base_url,
        "model_name": ua.model_name,
    }


def _sub_llm_config_from_ua(ua: UserAgent) -> dict[str, Any]:
    """
    打杂任务（记忆重写、知识库选档、RAG 打分/改写/HyDE）使用的子智能体 LLM 配置。
    仅当 sub_model_name 非空且子 Key 可解密出非空值时使用子配置（整体判定，不混搭）；
    否则实时回退主配置，主配置变更自动跟随。
    :param ua: 智能体
    :return: LLM 配置
    """
    plain = decrypt_api_key_safe(ua.sub_api_key_ciphertext)
    sub_model = (ua.sub_model_name or "").strip()
    if sub_model and (plain or "").strip():
        base_url = (ua.sub_base_url or "").strip() or None
        if base_url:
            from app.utils.ssrf import assert_public_http_url

            assert_public_http_url(base_url)
        return {
            "api_key": plain.strip(),
            "base_url": base_url,
            "model_name": sub_model,
        }
    return _llm_config_from_ua(ua)


def _run_kb_document_preselect_with_context(
    messages: list[BaseMessage],
    current_question: str,
    *,
    regenerate: bool,
    ua: UserAgent,
    agent_id: int,
) -> tuple[list[str] | None, dict[str, Any]]:
    """知识库前置选档：可附带最近多轮对话，meta 含 context_* 与选档结果。选档范围按属主隔离。"""
    from app.kb.kb_document_preselect import run_kb_document_preselect
    from app.kb.kb_preselect_context import build_kb_preselect_conversation_context
    from app.kb.kb_scope import kb_scope_for

    ctx, ctx_meta = build_kb_preselect_conversation_context(
        messages,
        user_text=(current_question or "").strip(),
        regenerate=regenerate,
    )
    filt, pre_meta = run_kb_document_preselect(
        (current_question or "").strip(),
        kb_scope_for(ua.user_id, agent_id),
        _sub_llm_config_from_ua(ua),
        conversation_context=ctx or None,
    )
    return filt, {**ctx_meta, **pre_meta}


def _mcp_approval_note(
    pending_id: str | None,
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
) -> str | None:
    if not (pending_id or "").strip():
        return None
    from app.mcp_client.tool_policy import peek_approved_mcp_call

    approved = peek_approved_mcp_call(
        pending_id, user_id=user_id, agent_id=agent_id, session_id=session_id
    )
    if not approved:
        return None
    return (
        f"用户已批准执行 MCP 工具 {approved.get('server_name')}/{approved.get('tool_name')} 一次。"
        "请立即调用该工具一次；不要调用其他高危工具。"
    )


def _prepare_to_invoke_messages(
    messages: list[BaseMessage],
    ua: UserAgent,
    user_id: int,
    agent_id: int,
    session_id: str,
    user_query_for_memory: str,
    *,
    use_knowledge_retrieval: bool,
    use_web_search: bool,
    document_filter: list[str] | None,
    session_attachment_hint: str,
    mcp_approval_note: str | None = None,
    path_ids: list[int] | None = None,
    image_caption: str | None = None,
    memory_inject_prefetched: str | None = None,
    skip_memory_retrieve: bool = False,
) -> list:
    """
    -所有预处理步骤的最后一个——把前面所有产物攒齐、组装成最终要喂给 Agent 的 messages-
    压缩或滑动窗口 → 可选会话记忆预检索 → 展开多模态（仅本轮 Human 带图）→ 本轮上下文并入最后一条 Human。
    path_ids 为 messages 对应的存储行 id（等长），供压缩与记忆按分支定位。
    image_caption 非空时（两阶段读图）：不再展开图片，描述随本轮上下文注入，agent 全程纯文本。
    """
    from app.chat.compact import build_compacted_model_messages
    from app.chat.memory_search import proactive_session_memory_inject_text
    from app.core.memory_turns import turn_keys_of
    from app.core.turn_runtime import emit_rag_step

    #步骤 1：压缩 or 滑动窗口
    llm_cfg = _sub_llm_config_from_ua(ua)
    if getattr(settings, "CHAT_USE_SESSION_MEMORY", True) and getattr(settings, "CHAT_COMPACT_ENABLED", True):
        #太长就压缩
        #内部判断：总字符超阈值 → 子智能体摘要 + 滑动窗口；不超 → 纯滑动窗口
        viewed = build_compacted_model_messages(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            llm_config=llm_cfg,
            system_chars=len(
                _compose_system_prompt(
                    ua,
                    use_knowledge_retrieval=use_knowledge_retrieval,
                    use_web_search=use_web_search,
                )
            ),
            path_ids=path_ids,
        )
    else:
        # 不压缩，（纯滑动窗口）
        viewed = apply_sliding_window_turns(messages)
    #步骤 2：主动预检索会话记忆
    # 用本轮用户问题去 Milvus 向量库捞 top 3 段相关的历史对话片段，返回格式化的文本（可直接拼进 System Prompt）
    if skip_memory_retrieve:
        inj = memory_inject_prefetched
    else:
        inj = proactive_session_memory_inject_text(
            (user_query_for_memory or "").strip(),
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            llm_config=llm_cfg,
            path_turn_keys=turn_keys_of(messages, path_ids) or None,
        )
    #步骤 3：展开多模态图片
    if inj:
        emit_rag_step("📌", "会话记忆预注入", "已附加较早轮次摘录")
    has_caption = bool((image_caption or "").strip())
    #图片变 data URL
    #把 messages 里的图片引用（image_ref）转成模型能看的格式
    expanded = expand_messages_for_model(
        viewed,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        images_on_last_human_only=not has_caption,
        expand_images=not has_caption,
    )
    #步骤 4：根据本轮规则，把本轮上下文并入最后一条 Human
    turn_block = _format_turn_context_block(
        use_knowledge_retrieval=use_knowledge_retrieval,
        use_web_search=use_web_search,
        document_filter=document_filter,
        session_attachment_hint=session_attachment_hint,
        memory_inject=inj,
        mcp_approval_note=mcp_approval_note,
        image_caption=image_caption,
    )
    return _append_turn_context_message(expanded, turn_block)


def _build_tools_legacy(
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    kb_scope: Any,
    llm_config: dict[str, Any],
    knowledge_base_document_filter: list[str] | None = None,
    extra_tools: list[Any] | None = None,
    workspace: Any = None,
    interrupt_on: dict[str, Any] | None = None,
) -> list[Any]:
    """旧的 9 段硬编码装配（U6b 之前）。

    **只为回退保留一个版本**：`CHAT_TOOL_REGISTRY_ENABLED=false` 时走这里。
    新工具不要加在这个函数里——它会在下个版本删除。新增工具改
    `app/chat/tool_providers/` 与 `app/chat/tool_registry.py`。
    """
    tools: list[Any] = []
    # ①会话附件工具
    tools.extend(make_session_attachment_tools(user_id, agent_id, session_id))
    # 网络搜索工具-联网三件套
    if getattr(settings, "WEB_SEARCH_ENABLED", True):
        tools.append(make_web_search_tool())  # ② 联网搜索
        tools.append(make_fetch_url_tool())  # ③ 获取 URL内容
        tools.append(make_web_image_search_tool())  # ④ 文字搜图
    # ⑤ 知识库检索，入口
    tools.append(
        make_search_knowledge_tool(
            kb_scope,
            llm_config,
            knowledge_base_document_filter=knowledge_base_document_filter,
        )
    )
    # ⑥以图检索
    tools.append(make_search_knowledge_by_image_tool(kb_scope, user_id, agent_id, session_id))
    # ⑦ 会话记忆工具
    if getattr(settings, "CHAT_USE_SESSION_MEMORY", True):
        tools.append(make_search_session_memory_tool(user_id, agent_id, session_id, llm_config))
    # ⑦b 跨会话长期记忆工具（U7）。与会话记忆并列而非替代：语义不同，配额也独立。
    from app.chat.memory_curation import user_scope_enabled

    if user_scope_enabled():
        from app.chat.memory_tool import make_search_long_term_memory_tool

        tools.append(make_search_long_term_memory_tool(user_id, agent_id))
    # ⑧ MCP 外部传入
    if extra_tools:
        tools.extend(extra_tools)
    # ⑨ 工作区工具（读/写/列目录）。命令执行工具只在「审批中断确实生效」时才注册——
    # 这个条件由 interrupt_on 里是否包含它推导，不让调用方各写一遍判断。
    if workspace is not None:
        from app.workspace.tools import make_workspace_tools

        allow_exec = bool(interrupt_on and interrupt_on.get("workspace_run_command"))
        tools.extend(make_workspace_tools(workspace, include_exec=allow_exec))
    return tools


def _sort_tools(tools: list[Any]) -> list[Any]:
    return sorted(tools, key=lambda t: str(getattr(t, "name", "") or ""))


def _approval_tools_map() -> dict[str, Any]:
    """把配置里的工具名列表转成 `interrupt_on` 需要的映射。

    `interrupt_on={name: True}` 表示允许 approve / edit / reject 三种决定。
    未列出的工具一律自动放行（LangChain 的默认语义）。
    """
    raw = str(getattr(settings, "CHAT_APPROVAL_TOOLS", "") or "")
    names = [x.strip() for x in raw.split(",") if x.strip()]
    return {n: True for n in names}


def _interrupt_to_event(payload: Any) -> dict[str, Any]:
    """把 LangGraph 的 `__interrupt__` 负载转成前端可渲染的扁平结构。

    原始结构是 `(Interrupt(value=HITLRequest(action_requests=[...],
    review_configs=[...]), id='...'),)`。前端不需要知道这么多层，
    这里拍平成 `{interrupt_id, actions, allowed_decisions}`。
    """
    items = payload if isinstance(payload, (list, tuple)) else [payload]
    actions: list[dict[str, Any]] = []
    decisions: list[str] = []
    interrupt_id: str | None = None

    for item in items:
        if interrupt_id is None:
            interrupt_id = getattr(item, "id", None)
        value = getattr(item, "value", None)
        if value is None and isinstance(item, dict):
            value = item
        if not isinstance(value, dict):
            continue
        for req in value.get("action_requests") or []:
            if not isinstance(req, dict):
                continue
            actions.append(
                {
                    "name": req.get("name"),
                    "args": req.get("args"),
                    "description": req.get("description"),
                }
            )
        for cfg in value.get("review_configs") or []:
            if isinstance(cfg, dict):
                for d in cfg.get("allowed_decisions") or []:
                    if d not in decisions:
                        decisions.append(d)

    return {
        "interrupt_id": interrupt_id,
        "actions": actions,
        "allowed_decisions": decisions or ["approve", "reject"],
    }


def _count_pending_actions(snapshot: Any) -> int:
    """数出断点里有多少个工具在等确认；取不到时返回 1。

    `HumanInTheLoopMiddleware.after_model` 强制校验
    `len(decisions) == len(interrupted_tool_calls)`，数量不足直接抛 ValueError。
    所以 resume 时必须按这个数填 `decisions`，不能固定填一个。

    返回 1 是保守兜底：绝大多数中断只有一个工具，退回旧行为不会让原本能跑的
    场景变坏；反之若猜大了，middleware 会因为数量多于 tool_call 数而报错。
    """
    try:
        for task in getattr(snapshot, "tasks", ()) or ():
            for intr in getattr(task, "interrupts", ()) or ():
                value = getattr(intr, "value", None)
                if not isinstance(value, dict):
                    continue
                n = len(value.get("action_requests") or ())
                if n:
                    return n
    except Exception:  # noqa: BLE001 - 数不出来就用默认值，不能让恢复本身失败
        logger.debug("统计待确认动作数失败（按 1 处理）", exc_info=True)
    return 1


def build_model_and_agent(
    ua: UserAgent,
    user_id: int,
    agent_id: int,
    session_id: str,
    *,
    knowledge_base_document_filter: list[str] | None = None,
    extra_tools: list[Any] | None = None,
    use_knowledge_retrieval: bool = True,
    use_web_search: bool = False,
    workspace: Any = None,
    checkpointer: Any = None,
    interrupt_on: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """
    构建模型和智能体。检索工具始终挂载（本轮禁用由工具函数返回 TOOL_DISABLED_THIS_TURN）。
    知识库检索按属主隔离：使用他人已发布智能体时检索发布者的知识库。

    :param workspace: 已打开的工作区实例（`app.workspace.BaseWorkspace`）；为 None 时
        不注册工作区工具。工作区由调用方负责创建与 cleanup——本函数只管用。
    :param checkpointer: LangGraph checkpointer；为 None 时无法中断/恢复
    :param interrupt_on: 需要人工审批的工具名映射（`{工具名: True}`）；为 None 时不启用中断
    """
    # 解密
    plain = decrypt_api_key_safe(ua.api_key_ciphertext)
    if not plain or not plain.strip():
        raise ValueError("智能体未配置有效的 API Key")

    base_url = (ua.base_url or "").strip() or None
    # SDK 层自动重试：只覆盖 429 / 5xx / 连接类瞬时故障（指数退避），鉴权与上下文超长不重试。
    # 此前主链路完全没有重试——一次限流就让整轮回答失败，用户只看到裸报错。
    retries_raw = getattr(settings, "LLM_MAX_RETRIES", 2)
    max_retries = max(0, int(retries_raw if retries_raw is not None else 2))
    model = LlmFactory.chat(
        {"api_key": plain, "base_url": base_url, "model_name": ua.model_name},
        temperature=float(ua.temperature),
        stream_usage=True,
        max_retries=max_retries,
    )
    model = _wrap_model_strip_images_after_tools(model)
    kb_scope = kb_scope_for(ua.user_id, ua.id)
    llm_config = _sub_llm_config_from_ua(ua)

    # 工具装配：默认走 provider 注册表（U6b），可用配置回退到旧路径一个版本。
    # provider 化之后，`agent_service` 不再知道任何具体工具——加工具只改
    # `app/chat/tool_providers/` 与 `tool_registry.TOOL_SPECS` 两处。
    prompt_fragments: list[str] | None = None
    if bool(getattr(settings, "CHAT_TOOL_REGISTRY_ENABLED", True)):
        from app.chat.tool_providers import ToolBuildContext, build_tools

        tools, prompt_fragments = build_tools(
            ToolBuildContext(
                ua=ua,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                kb_scope=kb_scope,
                llm_config=llm_config,
                workspace=workspace,
                interrupt_on=interrupt_on,
                extra_tools=list(extra_tools or []),
                knowledge_base_document_filter=knowledge_base_document_filter,
                use_knowledge_retrieval=use_knowledge_retrieval,
                use_web_search=use_web_search,
            )
        )
    else:
        tools = _build_tools_legacy(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            kb_scope=kb_scope,
            llm_config=llm_config,
            knowledge_base_document_filter=knowledge_base_document_filter,
            extra_tools=extra_tools,
            workspace=workspace,
            interrupt_on=interrupt_on,
        )

    # 工具按名称排序（顺序稳定，有利于 prompt 前缀缓存命中）
    tools = _sort_tools(tools)
    # 注册表校验：重名必须抛错——LangChain 对重名工具是**静默保留其一**（已实测），
    # 而 MCP 顶替 workspace_write_file / run_command 意味着绕过容器隔离。
    # 未登记只告警：MCP 工具是动态的，本就不在 TOOL_SPECS 里。
    from app.chat.tool_registry import assert_no_duplicates, warn_unknown_tools

    assert_no_duplicates(tools)
    warn_unknown_tools(
        tools, exempt=[str(getattr(t, "name", "") or "") for t in (extra_tools or [])]
    )

    from langchain.agents import create_agent
    # 智能体人设提示词
    system_prompt = _compose_system_prompt(
        ua,
        use_knowledge_retrieval=use_knowledge_retrieval,
        use_web_search=use_web_search,
        fragments=prompt_fragments,
    )
    # 智能体参数
    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "system_prompt": system_prompt,
    }
    # checkpointer 是中断/恢复的前提：没有它，interrupt 无处落盘，也就无法 resume
    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer

    # 中间件链：图片过滤（可选）+ 人工审批（可选）。
    # 两者都在 middleware 里，顺序不冲突——strip_images 处理模型输入，
    # HITL 在 after_model 拦截 tool_calls，互不干涉。
    middlewares: list[Any] = []
    strip_mw = _try_strip_images_middleware()
    if strip_mw is not None:
        middlewares.append(strip_mw)
    if interrupt_on and checkpointer is not None:
        try:
            from langchain.agents.middleware import HumanInTheLoopMiddleware

            middlewares.append(
                HumanInTheLoopMiddleware(
                    interrupt_on=interrupt_on,
                    description_prefix="以下工具需要你的确认才会执行",
                )
            )
        except Exception as e:  # noqa: BLE001
            # 审批中间件装不上时**必须显式降级**，不能静默：用户会以为有保护，实际没有
            logger.warning("审批中间件加载失败，本轮无人工确认（高危工具将被直接拒绝）：{}", e)
            agent_kwargs.pop("checkpointer", None)
            middlewares = [strip_mw] if strip_mw is not None else []

    # 规划（U9a）：给模型 `write_todos` 工具 + 规划提示词，todos 进 agent state。
    # 必须放在 HITL 之后——上面那段审批失败时会把 middlewares 整体重置，
    # 提前 append 会被一起抹掉（这是这段代码的既有坑，别把顺序调回去）。
    if _todo_enabled():
        try:
            from langchain.agents.middleware import TodoListMiddleware

            middlewares.append(
                TodoListMiddleware(
                    system_prompt=_TODO_SYSTEM_PROMPT,
                    tool_description=_TODO_TOOL_DESCRIPTION,
                )
            )
        except Exception as e:  # noqa: BLE001 - 装不上就本轮没有清单，不阻断对话
            logger.warning("规划中间件加载失败（本轮无待办清单）：{}", e)

    agent = create_agent(**agent_kwargs, middleware=middlewares or None)
    return agent, model


def _todo_enabled() -> bool:
    """U9a 规划开关。关闭后模型没有 `write_todos`，提示词里也没有规划段。"""
    return bool(getattr(settings, "CHAT_TODO_ENABLED", True))


# 为什么覆盖上游提示词（2026-10-24 实测）：
# 上游 `WRITE_TODOS_SYSTEM_PROMPT` 的措辞是"可选"——"use it when it is helpful"
# "for simple objectives it is better to just complete the objective directly"。
# deepseek 读成了"锦上添花"，面对"写文档+配套练习文件"这种明确多步任务也没调用
# （线上实录：一轮里检索、生成都有，todo_list 事件 0 条）。
# 这里把触发条件写成**硬条件**，并要求清单对用户可读——它是要展示给人看的。
_TODO_SYSTEM_PROMPT = """## 任务清单（write_todos）

你有 `write_todos` 工具，用来把多步任务拆成清单，并让用户实时看到进度。

**满足以下任一条，就必须在动手前先写清单**：
1. 任务需要 3 个及以上步骤；
2. 需要调用 2 个及以上工具（知识库检索 / 联网搜索 / 读写文件 / 执行命令）；
3. 用户一次给了多件事（编号、分号或"然后/再"串起来的多项要求）。
只有"一句话就能答完"的问题才不用清单。

**清单怎么维护**：
- 开始做某一步之前，把该步标成 `in_progress`；
- 该步做完立刻标 `completed`，不要攒到最后一起改；
- 中途发现新步骤就补进去，做不了的旧步骤可以删，但已完成的不许改回去；
- 一条一句话，用简体中文写清楚"要做什么/做完了什么"。

清单会直接展示给用户，所以要写具体动作（例如"检索知识库中的建表语法"），
不要写"思考中""处理中"这类空话。"""

_TODO_TOOL_DESCRIPTION = (
    "为当前任务创建/更新结构化清单，用户会实时看到每一项的状态。"
    "任务有 3 步以上、或需要调用 2 个以上工具时，动手前必须先调用本工具列计划；"
    "之后每完成一步就更新状态。整体替换语义：每次传入完整清单。"
)


# 前端契约只认这三态；模型写出别的值时按 pending 兜底（不往前端丢未知状态）
_TODO_STATUSES = frozenset({"pending", "in_progress", "completed"})


def _normalize_todos(raw: Any) -> list[dict] | None:
    """把中间件写进 state 的 todos 收敛成前端契约的结构。

    防御模型乱写：content 必须是非空字符串，status 不在三态内一律按 pending。
    返回 None 表示「这次更新里没有可用的 todos」——注意与「空清单」的区别：
    空清单是该轮到没有待办，None 是这次 payload 跟我们无关。
    """
    if not isinstance(raw, (list, tuple)):
        return None
    items: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        status = str(item.get("status") or "").strip().lower()
        items.append(
            {
                "content": content,
                "status": status if status in _TODO_STATUSES else "pending",
            }
        )
    return items or None


def _extract_todos(payload: Any) -> list[dict] | None:
    """从 updates 流的一个 payload 里取出 todos。

    实测形状（2026-10-24，langchain 1.2.15，见 docs/dev/U9-思路.md §2.1）：
        {"tools": {"todos": [...], "messages": [...]}}
    `write_todos` 返回 `Command(update={"todos": ...})`，该更新归到 tools 节点。
    不同版本可能换节点名，所以按「值里有没有 todos 键」找，不写死 "tools"。
    """
    if not isinstance(payload, dict):
        return None
    for update in payload.values():
        if isinstance(update, dict) and "todos" in update:
            return _normalize_todos(update.get("todos"))
    return None




def _throttled_cancel_check(
    cancel_check: Callable[[], bool] | None,
) -> Callable[[], Any] | None:
    """
    取消检查节流包装：流式循环按 chunk 频率调用本函数，但实际读 Redis（线程跳转）
    最多每 CHAT_CANCEL_CHECK_INTERVAL 秒一次，期间返回上次结果。
    一旦观察到取消则永久为真（取消不可逆）。
    """
    if cancel_check is None:
        return None
    interval = max(0.05, float(getattr(settings, "CHAT_CANCEL_CHECK_INTERVAL", 0.25) or 0.25))
    state = {"last_at": 0.0, "flag": False}

    async def _probe() -> bool:
        if state["flag"]:
            return True
        now = time.monotonic()
        if now - state["last_at"] < interval:  #0.25s 内不重复问 Redis
            return False
        state["last_at"] = now
        try:
            state["flag"] = bool(await asyncio.to_thread(cancel_check))
        except Exception:
            state["flag"] = False
        return state["flag"]

    return _probe


# ============================================================================
# 对话编排的零件（拆分自原先 757 行的 iter_chat_stream_events）
#
# 拆分原则：一个函数只负责一件事；事件的**顺序**与状态变更逐行保持原样。
# 编排函数按 ①→⑮ 的阶段顺序读下来即可，不必在脑子里执行 757 行。
# 等价性由 tests/chat/test_agent_stream_contract.py 的 12 条契约测试守着
# （普通对话 / 流中取消 / 启动前取消 / 审批中断 / resume 三种非法 /
#   regenerate 两路 / MCP 续跑 / 模型报错 / trace+sources+MCP 事件）。
# ============================================================================


class _ThinkingCollector:
    """思考区与正文事件的统一出口。

    拆分前这是四个必须同步更新的东西：`output_queue`（实时事件）、
    `rag_steps_collected`（落库步骤）、`thinking_items_collected`（落库时间线）、
    `thinking_text_parts`（落库过渡文本），外加一个 `_RagStepProxy` 把 RAG 步骤
    同时塞进前三处。分散着写极易漏一处，而漏了的表现是「实时看得到、刷新就没了」。

    注意有两类「步骤/文本」写法，行为不同，别混用：
    - `add_step`：进队列 → 由编排函数统一出队推给前端（检索/工具步骤走这条）
    - `add_inline_step`：**不进队列**，直接返回事件由编排函数 yield（图片预读走这条；
      若也进队列，会被排到所有正文之后，顺序就错了）
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.rag_steps: list[dict] = []
        self.thinking_items: list[dict] = []
        self.thinking_text_parts: list[str] = []
        self.full_response = ""
        self.image_caption: str | None = None
        # 最近一次待办清单（收尾时落库用；实时变化走 SSE，不进时间线）
        self.latest_todos: list[dict] | None = None

    def set_todos(self, todos: list[dict] | None) -> None:
        """更新待办清单并推事件。

        内容没变就不推：模型经常在同一个阶段反复写同样的表，逐次推送只会让前端白重渲染。
        """
        if not todos or todos == self.latest_todos:
            return
        self.latest_todos = todos
        self.queue.put_nowait({"type": "todo_list", "todos": list(todos)})

    def add_step(self, icon: str, label: str, detail: str = "") -> None:
        """写一条步骤：实时事件 + 落库步骤 + 落库时间线，三处同步。"""
        step = {"icon": icon, "label": label, "detail": detail}
        self.rag_steps.append(step)
        item = {"type": "step", **step}
        self.thinking_items.append(item)
        self.queue.put_nowait({"type": "thinking_item", "item": item})

    def add_inline_step(self, icon: str, label: str, detail: str = "") -> dict:
        """同 `add_step`，但**不进队列**：返回事件给调用方直接 yield。"""
        step = {"icon": icon, "label": label, "detail": detail}
        self.rag_steps.append(step)
        item = {"type": "step", **step}
        self.thinking_items.append(item)
        return {"type": "thinking_item", "item": item}

    def merge_text(self, text: str) -> None:
        """把文本并入思考区时间线（相邻文本合并，避免碎成一堆单项）。

        只动时间线，不计入 `thinking_text`，也不产生事件——图片预读的流式描述
        由调用方自己 yield，且它不算「工具调用前的过渡文本」。
        """
        if self.thinking_items and self.thinking_items[-1].get("type") == "text":
            self.thinking_items[-1]["text"] += text
        else:
            self.thinking_items.append({"type": "text", "text": text})

    def push_transition_text(self, text: str) -> None:
        """工具调用**之后**继续流出的正文：并入时间线并追加事件。

        与 `move_transition_text` 的差别：这里合并到上一项（append=True），
        那里是工具前导句，必须新起一项并标记 `moved_from_content`。
        """
        self.thinking_text_parts.append(text)
        self.merge_text(text)
        self.queue.put_nowait(
            {
                "type": "thinking_item",
                "item": {"type": "text", "text": text},
                "append": True,
            }
        )

    def move_transition_text(self, text: str) -> dict:
        """工具调用**之前**已当正文发出去的那段文本：从正文撤回、迁入思考区。

        :return: 待发事件（`moved_from_content=True` 告诉前端把已渲染的正文挪走）
        """
        self.thinking_text_parts.append(text)
        item = {"type": "text", "text": text}
        self.thinking_items.append(item)
        if self.full_response.endswith(text):
            self.full_response = self.full_response[: -len(text)]
        else:
            pos = self.full_response.rfind(text)
            self.full_response = self.full_response[:pos] if pos >= 0 else self.full_response
        return {"type": "thinking_item", "item": item, "moved_from_content": True}

    def bind_rag_step_queue(self) -> None:
        """注册为 `emit_rag_step` 的落点（检索/工具步骤从别处打进来）。"""
        collector = self

        class _Sink:
            def put_nowait(self, step: dict) -> None:
                collector.rag_steps.append(step)
                item = {"type": "step", **step}
                collector.thinking_items.append(item)
                collector.queue.put_nowait({"type": "thinking_item", "item": item})

        set_rag_step_queue(_Sink())

    def drain(self) -> list[dict]:
        """取出并清空当前已排队的事件（保持入队顺序）。"""
        events: list[dict] = []
        while not self.queue.empty():
            events.append(self.queue.get_nowait())
        return events


class _StreamOutcome:
    """一轮生成的终态。拆 `_run_agent_stream` 时用它替代 5 个 nonlocal 变量。"""

    __slots__ = ("error", "cancelled", "interrupted", "interrupt_event")

    def __init__(self) -> None:
        self.error: str | None = None
        self.cancelled = False
        self.interrupted = False
        self.interrupt_event: dict[str, Any] | None = None


@dataclass
class _TurnInputs:
    """`is_resume` / `regenerate` / 普通提问 三路归一后的输入。"""

    messages: list[BaseMessage]
    preselect_query: str
    regen_target_ai_id: int | None = None
    regen_path_ids: list[int] | None = None
    error: str | None = None


def _resolve_turn_inputs(
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    user_text: str,
    is_resume: bool,
    regenerate: bool,
    target_message_id: int | None,
    resume_thread_id: str | None,
) -> _TurnInputs:
    """定出「这一轮基于哪些历史、重写哪条消息」。

    三种输入的分支不能合并：regenerate 会用 `get_regenerate_context` 把历史重建成
    模型输入，而 resume 的输入完全由 checkpoint 提供（`Command(resume=...)`），
    重建会与断点状态打架。因此 resume 只借用「按 target_message_id 定位一条助手消息」。
    """
    if is_resume:
        # ★ 属主校验（安全边界，不能省）：
        # `resume_thread_id` 是**用户可控输入**。若只按它去 recover checkpoint，
        # 任何人拿到别人的 job_id 就能恢复别人的会话——看到别人的对话内容，
        # 甚至触发对方被中断的高危工具执行（实测确实会执行）。
        # 这里用「目标消息必须属于本会话，且其 approval_pending.thread_id 与请求一致」
        # 双重约束：既挡住跨用户，也挡住跨会话/伪造 thread。
        path_records = storage.get_session_messages(user_id, agent_id, session_id)
        target_rec = next(
            (r for r in path_records if int(r.get("message_id") or 0) == int(target_message_id or 0)),
            None,
        )
        pending = (target_rec or {}).get("approval_pending") or {}
        if target_rec is None or not pending or pending.get("thread_id") != resume_thread_id:
            return _TurnInputs(messages=[], preselect_query="", error="审批已失效或不属于当前会话，请重新发起提问")
        # messages 仍要加载：后续预选/记忆逻辑会读它，只是不会作为模型输入
        return _TurnInputs(
            messages=storage.load(user_id, agent_id, session_id),
            preselect_query=(user_text or "").strip(),
            regen_target_ai_id=int(target_message_id or 0),
        )

    if regenerate:
        regen_target_ai_id: int | None = None
        if target_message_id:
            regen_target_ai_id = int(target_message_id)
        else:
            path_records = storage.get_session_messages(user_id, agent_id, session_id)
            target_rec = next((r for r in reversed(path_records) if r.get("type") == "ai"), None)
            if target_rec is not None:
                regen_target_ai_id = int(target_rec.get("message_id") or 0) or None
        if not regen_target_ai_id:
            return _TurnInputs(messages=[], preselect_query="", error="没有可重新生成的对话")

        ctx = storage.get_regenerate_context(user_id, agent_id, session_id, regen_target_ai_id)
        if ctx is None:
            return _TurnInputs(
                messages=[], preselect_query="", error="无法重新生成：目标回复不存在或不属于当前会话"
            )
        messages, last_human_text, regen_path_ids = ctx
        req_text = (user_text or "").strip()
        if req_text and req_text != last_human_text.strip():
            return _TurnInputs(
                messages=[], preselect_query="", error="重新生成失败：请求文案与目标提问不一致，请刷新后重试"
            )
        return _TurnInputs(
            messages=messages,
            preselect_query=msg_content_to_str(messages[-1].content).strip(),
            regen_target_ai_id=regen_target_ai_id,
            regen_path_ids=regen_path_ids,
        )

    return _TurnInputs(
        messages=storage.load(user_id, agent_id, session_id),
        preselect_query=(user_text or "").strip(),
    )


async def _prefetch_kb_documents(
    ua: UserAgent,
    messages: list[BaseMessage],
    preselect_query: str,
    *,
    agent_id: int,
    regenerate: bool,
    enabled: bool,
) -> tuple[list[str] | None, dict]:
    """知识库选档：本轮该只在哪些文档里检索。"""
    if not enabled:
        return None, {}
    return await asyncio.to_thread(
        partial(
            _run_kb_document_preselect_with_context,
            messages,
            preselect_query,
            regenerate=regenerate,
            ua=ua,
            agent_id=agent_id,
        )
    )


async def _prefetch_session_memory(
    ua: UserAgent,
    preselect_query: str,
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
) -> str | None:
    """主动注入 System 的记忆文本。

    U7 起是**两段**拼起来的：
    1. 会话级（本会话较早的原话摘录）—— 受 `CHAT_USE_SESSION_MEMORY` 控制
    2. 用户级（以往会话提炼出的稳定事实）—— 受 `CHAT_MEMORY_*` 控制

    两段各自独立取，任何一段为空都不影响另一段——这很重要：会话级依赖 compact
    触发（苛刻），用户级依赖 curation（宽松），混在一个条件里会让用户级被会话级的
    "还没到阈值" 拖累成空。
    """
    from app.chat.memory_search import proactive_session_memory_inject_text

    # 会话级
    session_part: str | None = None
    if getattr(settings, "CHAT_USE_SESSION_MEMORY", True) and getattr(
        settings, "CHAT_MEMORY_PROACTIVE_INJECT", True
    ):
        llm_cfg = _sub_llm_config_from_ua(ua)
        session_part = await asyncio.to_thread(
            proactive_session_memory_inject_text,
            preselect_query,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            llm_config=llm_cfg,
        )

    # 用户级（跨会话）
    user_part: str | None = None
    from app.chat.memory_curation import user_scope_enabled

    if user_scope_enabled():
        from app.chat.memory_search import (
            format_user_memory_hits,
            retrieve_user_memory_hits,
        )

        top_k = max(1, int(getattr(settings, "CHAT_MEMORY_USER_SCOPE_INJECT_TOP_K", 3) or 3))
        hits = await asyncio.to_thread(
            retrieve_user_memory_hits,
            preselect_query,
            user_id=user_id,
            agent_id=agent_id,
            top_k=top_k,
        )
        if hits:
            user_part = format_user_memory_hits(hits)

    if session_part and user_part:
        return f"{user_part}\n\n{session_part}"
    return user_part or session_part


async def _open_workspace_for_turn(
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    job_id: str | None,
) -> Any:
    """开工作区（容器/Local）。

    容器启动要 1–2s，所以与 KB/记忆预取并行；不可用时返回 None，
    本轮只是没有文件工具，对话照常。
    """
    from app.workspace import open_workspace

    try:
        return await asyncio.to_thread(
            open_workspace,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            # /chat/stream 没有 job 概念，退化成一个随机短标识（仅用于容器命名）
            job_id=job_id or f"s{uuid.uuid4().hex[:12]}",
        )
    except Exception as e:  # noqa: BLE001 - 工作区失败绝不能中断对话
        logger.warning("工作区初始化失败（本轮无文件能力）: {}", e)
        return None


async def _prefetch_turn_side_effects(
    ua: UserAgent,
    messages: list[BaseMessage],
    preselect_query: str,
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    regenerate: bool,
    use_knowledge_retrieval: bool,
    job_id: str | None,
) -> tuple[list[str] | None, dict, str | None, Any]:
    """三路并行预取：知识库选档 / 会话记忆 / 工作区。"""
    (retrieval_filter, kb_preselect_meta), memory_text, workspace = await asyncio.gather(
        _prefetch_kb_documents(
            ua,
            messages,
            preselect_query,
            agent_id=agent_id,
            regenerate=regenerate,
            enabled=use_knowledge_retrieval,
        ),
        _prefetch_session_memory(
            ua, preselect_query, user_id=user_id, agent_id=agent_id, session_id=session_id
        ),
        _open_workspace_for_turn(
            user_id=user_id, agent_id=agent_id, session_id=session_id, job_id=job_id
        ),
    )
    return retrieval_filter, kb_preselect_meta, memory_text, workspace


async def _resolve_checkpointer(*, thread_id: str | None) -> tuple[Any, dict[str, Any]]:
    """取 checkpointer 与「哪些工具要人工确认」。

    只有带 job_id 的路径（/chat/jobs）才启用：/chat/stream 直连流式没有 job 概念，
    也就没有 thread_id，中断无处落盘——那条路径保持原行为（高危工具不注册即无法执行）。

    resume 时线程标识必须是**原 job**：checkpoint 绑定在中断那一轮的 job_id 上，
    用户点确认产生的是新 job（新 id），所以要由前端把原 id 带回来。
    """
    if not thread_id:
        return None, {}
    try:
        from app.chat.checkpoint import get_checkpointer

        checkpointer = await get_checkpointer()
    except Exception as e:  # noqa: BLE001
        logger.warning("checkpoint 获取失败，本轮无审批中断：{}", e)
        return None, {}
    if checkpointer is None:
        return None, {}
    return checkpointer, _approval_tools_map()


async def _verify_resume_snapshot(
    agent: Any,
    *,
    thread_id: str | None,
    is_resume: bool,
    checkpointer: Any,
) -> tuple[int, str | None]:
    """resume 前确认「断点还在」，并数出本次有几个工具在等确认。

    没有这一层，给一个不存在的 thread 传 `Command(resume=...)` 时 langgraph 会把它
    当成普通输入**从头跑一整轮**——用户以为在恢复审批，实际是重跑了一次对话，
    且表面上一切正常、没有任何报错（实测确认）。

    顺手数出待确认动作数：HITL middleware 强制
    `len(decisions) == len(interrupted_tool_calls)`，少一个就抛 ValueError；
    模型一次要调两个需审批工具时，只传 1 个决定会让恢复直接失败。
    """
    if not is_resume or checkpointer is None:
        return 1, None
    try:
        snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    except Exception:  # noqa: BLE001
        snapshot = None
    if snapshot is None or not getattr(snapshot, "next", None):
        return 1, "审批已过期或已完成，请重新发起提问"
    return _count_pending_actions(snapshot), None


def _chunk_text(msg: AIMessageChunk) -> str:
    """从 chunk 里取纯文本（兼容 str 与 OpenAI 的 block 列表两种形状）。"""
    if isinstance(msg.content, str):
        return msg.content
    if not isinstance(msg.content, list):
        return ""
    text = ""
    for block in msg.content:
        if isinstance(block, str):
            text += block
        elif isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text", "")
    return text


async def _run_agent_stream(
    agent: Any,
    stream_input: Any,
    collector: _ThinkingCollector,
    outcome: _StreamOutcome,
    cancel_probe: Callable[[], Any] | None,
    *,
    use_updates: bool,
    thread_id: str | None,
) -> None:
    """跑 agent 的流，把 chunk 分成「正文」与「思考区」两路推入队列。

    分流规则（原实现的核心不变量）：
    - 一条消息先按正文流出；一旦它出现 `tool_call_chunks`，此前已流的正文要撤回
      并迁入思考区（那是工具前导句，不是回答），此后同一消息的文本一律进思考区。
    - 换消息（`msg.id` 变化）就重置该判定。
    """
    current_msg_id: str | None = None
    msg_text_emitted = ""  # 当前消息已按 content 发出的文本
    msg_moved = False  # 当前消息已被判定为工具调用，剩余文本直发 thinking_text
    queue = collector.queue
    try:
        stream = agent.astream(
            stream_input,
            stream_mode=["messages", "updates"] if use_updates else "messages",
            config=_agent_invoke_config(thread_id if use_updates else None),
        )
        async for item in stream:
            if cancel_probe is not None and await cancel_probe():
                outcome.cancelled = True
                break

            if use_updates:
                # 多 stream_mode 时输出是 (mode, payload)；messages 模式下
                # payload 才是 (chunk, metadata)，其余保持原解包逻辑
                mode, payload = item
                if mode == "updates":
                    interrupt_payload = (
                        payload.get("__interrupt__") if isinstance(payload, dict) else None
                    )
                    if interrupt_payload:
                        outcome.interrupted = True
                        outcome.interrupt_event = _interrupt_to_event(interrupt_payload)
                        await queue.put(
                            {
                                "type": "approval_required",
                                "request": outcome.interrupt_event,
                                # 前端恢复时必须知道「恢复哪个 checkpoint」。
                                # 落库的消息里也有 thread_id，但实时路径拿不到那条记录
                                # （它在本轮收尾才写入），所以事件里必须显式带上。
                                "thread_id": thread_id,
                            }
                        )
                        # 中断是正常暂停，不是错误：这里必须跳出循环结束本轮，
                        # 否则后续会误落一条空的助手消息
                        break
                    # 规划：write_todos 的更新归到 tools 节点，形状见 _extract_todos
                    collector.set_todos(_extract_todos(payload))
                    continue
                msg, _metadata = payload
            else:
                msg, _metadata = item

            if not isinstance(msg, AIMessageChunk):
                continue

            content = _chunk_text(msg)
            has_tool_call = bool(getattr(msg, "tool_call_chunks", None))
            # 新消息重置状态
            if msg.id is not None and msg.id != current_msg_id:
                current_msg_id = msg.id
                msg_text_emitted = ""
                msg_moved = False

            if has_tool_call:
                if msg_text_emitted:
                    # 本消息此前作为正文流出的文本实为工具调用前导句：移交前端迁入思考区
                    queue.put_nowait(collector.move_transition_text(msg_text_emitted))
                    msg_text_emitted = ""
                if msg.id is not None:
                    msg_moved = True

            if content:
                if msg_moved:
                    collector.push_transition_text(content)
                else:
                    collector.full_response += content
                    if msg.id is not None:
                        msg_text_emitted += content
                    queue.put_nowait({"type": "content", "content": content})
    except asyncio.CancelledError:
        outcome.cancelled = True
        raise
    except Exception as e:
        # 分类后再出口：直接抛 str(e) 会把 provider 原始报文（含 base_url、Key 片段）送到前端，
        # 且用户无法判断该重试还是该改配置。error_text 落库存 detail，事件只给用户文案。
        info = classify_llm_error(e)
        # 落库的 error_text 会被前端渲染进历史消息（useChatMessages.js 取 row.error_text），
        # 所以这里必须存用户文案；原始报文含 base_url / Key 片段，不上行也不入库。
        outcome.error = info.user_message
        await queue.put(
            {
                "type": "error",
                "content": info.user_message,
                "kind": info.kind,
                "retryable": info.retryable,
            }
        )
    finally:
        await queue.put(None)


async def _cancel_watchdog(agent_task: asyncio.Task, cancel_probe: Callable[[], Any] | None) -> None:
    """（0.3 秒周期）轮询取消标记，命中即中断生成任务。

    覆盖工具执行等「没有 chunk 流出」的阶段——chunk 循环里的取消检查在这些阶段不会触发。
    """
    if cancel_probe is None:
        return
    while not agent_task.done():
        if await cancel_probe():
            agent_task.cancel()
            return
        await asyncio.sleep(0.3)


async def _run_vision_prepass(
    ua: UserAgent,
    messages: list[BaseMessage],
    last_human_msg: BaseMessage,
    collector: _ThinkingCollector,
    outcome: _StreamOutcome,
    cancel_probe: Callable[[], Any] | None,
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    use_knowledge_retrieval: bool,
    use_web_search: bool,
    has_mcp_tools: bool,
) -> AsyncIterator[dict]:
    """图片预读：先让视觉模型把图读成文字，主模型就不必真看像素。

    取消时置 `outcome.cancelled` 并结束迭代；清理与 cancelled/done 事件由编排函数补发
    （这里 yield 的事件必须「直接出」，不能进队列，否则会被排到所有正文之后）。
    """
    yield collector.add_inline_step("🖼️", "正在理解图片", "已附加图片，等待模型分析")

    if not _should_run_vision_caption(
        messages,
        ua,
        use_knowledge_retrieval=use_knowledge_retrieval,
        use_web_search=use_web_search,
        has_mcp_tools=has_mcp_tools,
    ):
        return

    from app.chat.vision_caption import iter_image_caption_chunks

    caption_parts: list[str] = []
    caption_ok = False
    try:
        async for text in iter_image_caption_chunks(
            ua,
            last_human_msg,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        ):
            if cancel_probe is not None and await cancel_probe():
                outcome.cancelled = True
                return
            caption_parts.append(text)
            collector.merge_text(text)
            yield {
                "type": "thinking_item",
                "item": {"type": "text", "text": text},
                "append": True,
            }
        caption_ok = True
    except Exception:  # noqa: BLE001 - 读图失败只降级，不影响问答
        caption_ok = False

    caption = "".join(caption_parts).strip()
    if caption_ok and caption:
        collector.image_caption = caption
        yield collector.add_inline_step("🖼️", "图片理解完成", f"已预读图片内容（{len(caption)} 字）")
    else:
        yield collector.add_inline_step("⚠️", "图片理解失败", "改为直接带图问答")


async def _finish_interrupted(
    collector: _ThinkingCollector,
    outcome: _StreamOutcome,
    collector_ref: tuple[int, int, str],
    *,
    thread_id: str | None,
    is_resume: bool = False,
    regen_target_ai_id: int | None = None,
) -> AsyncIterator[dict]:
    """审批中断的收尾：落一条「待确认」助手消息，并把它的 id 带回前端。"""
    user_id, agent_id, session_id = collector_ref
    # 审批中断：图停在中途，等用户决定。
    #
    # 必须落一条助手消息（`approval_pending` 标记）：否则用户刷新页面后确认卡片
    # 就消失了，而且 resume 时无从定位该更新哪一行（会误伤上一轮回答）。
    # 这不是「最终回答」，但它是这次交互的事实记录，与 U2a 的快照思路一致。
    #
    # 思考区也要一起存：只存 approval_pending 的话，刷新后卡片在、步骤与清单全没了
    # （用户看到的是「只剩一个待确认框」，不知道模型做到哪一步）。
    if collector.latest_todos:
        collector.thinking_items.append({"type": "todo", "todos": collector.latest_todos})
    pending_extra = {
        "approval_pending": {
            **(outcome.interrupt_event or {}),
            "thread_id": thread_id,
        },
        "error_text": None,
        "rag_steps": collector.rag_steps or None,
        "thinking_text": "".join(collector.thinking_text_parts) or None,
        "thinking_items": collector.thinking_items or None,
    }
    ai_msg = AIMessage(content=collector.full_response or "")

    # ★ resume 后**又**中断时不能新增一行：那会留下上一张已经失效的卡片。
    # 用户的真实路径是「批准 → 模型继续 → 再次要求高危命令 → 再中断」，
    # 每次新增一行，会话里就堆一串待确认消息，旧卡片的 interrupt 早已不存在，
    # 点上去只会得到「审批已失效」。所以沿用 resume 完成路径的原地覆盖语义。
    saved = False
    if is_resume and regen_target_ai_id is not None:
        try:
            saved = storage.update_assistant_in_place(
                user_id, agent_id, session_id, regen_target_ai_id, ai_msg, extra=pending_extra
            )
        except Exception:  # noqa: BLE001 - 覆盖失败就退回追加，别让中断事件发不出去
            logger.warning("覆盖待确认消息失败，改为追加新行", exc_info=True)
            saved = False
    if not saved:
        try:
            await asyncio.to_thread(
                storage.append_messages,
                user_id,
                agent_id,
                session_id,
                [ai_msg],
                extra_message_data=[pending_extra],
            )
        except Exception:  # noqa: BLE001 - 落库失败也要把中断事件送出去
            logger.warning("落库待确认消息失败（不影响前端确认卡片）", exc_info=True)

    # 把刚落库那条消息的 id 带回给前端：恢复时前端需要它作 target_message_id。
    # 不给的话前端只能再拉一次会话列表，而且此刻消息可能还没写完，时序上不可靠。
    pending_message_id: int | None = None
    try:
        recs = await asyncio.to_thread(storage.get_session_messages, user_id, agent_id, session_id)
        last_ai = next(
            (
                r
                for r in reversed(recs)
                if r.get("type") == "ai" and (r.get("approval_pending") or {}).get("thread_id")
            ),
            None,
        )
        if last_ai is not None:
            pending_message_id = int(last_ai.get("message_id") or 0) or None
    except Exception:  # noqa: BLE001
        logger.warning("查询待确认消息 id 失败（前端需刷新获取）", exc_info=True)

    get_last_rag_context(clear=True)
    get_pending_mcp_confirmations(clear=True)
    yield {
        "type": "done",
        "cancelled": False,
        "interrupted": True,
        "message_id": pending_message_id,
    }


async def _finish_completed(
    collector: _ThinkingCollector,
    outcome: _StreamOutcome,
    messages: list[BaseMessage],
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    is_resume: bool,
    regenerate: bool,
    regen_target_ai_id: int | None,
    mcp_approved_pending_id: str | None,
    kb_preselect_meta: dict,
    llm_config: dict | None = None,
) -> AsyncIterator[dict]:
    """正常收尾：取检索上下文 → 发 trace/sources/待确认 → 落库 → 归档记忆 → 提炼长期记忆。"""
    # 获取 RAG 上下文（顺带清空，避免串到下一轮）
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None
    pending_mcp = get_pending_mcp_confirmations(clear=True)
    image_references = rag_context.get("image_references") if rag_context else None
    # 知识库 / 联网搜索互斥，合并走统一 sources 通道
    kb_sources = rag_context.get("kb_sources") if rag_context else None
    web_sources = rag_context.get("web_sources") if rag_context else None
    merged_sources = kb_sources or web_sources

    if rag_trace:
        yield {"type": "trace", "rag_trace": rag_trace}

    if merged_sources:
        yield {"type": "sources", "sources": merged_sources}

    for pending in pending_mcp:
        yield {"type": "mcp_confirmation_required", "confirmation": pending}

    # 规划清单只存终态一份：实时过程走 SSE `todo_list`，历史回放要的是「最后做完的样子」。
    # 每次变化都追加会把时间线刷成十几条快照，和「思考步骤」混在一起。
    if collector.latest_todos:
        collector.thinking_items.append({"type": "todo", "todos": collector.latest_todos})

    ai_msg = AIMessage(content=collector.full_response)
    messages.append(ai_msg)
    ai_extra = {
        "rag_trace": rag_trace,
        "rag_steps": collector.rag_steps or None,
        "error_text": outcome.error,
        "image_references": image_references,
        "sources": merged_sources,
        "kb_preselect": kb_preselect_meta or None,
        "thinking_text": "".join(collector.thinking_text_parts) or None,
        "thinking_items": collector.thinking_items or None,
    }

    if is_resume:
        # 恢复：原地更新中断时那条「待确认」消息，不新增行——
        # 否则历史里会留下一条空壳 + 一条正式回答，用户看到的是一次提问两条回复
        saved = False
        if regen_target_ai_id is not None:
            # merge_thinking=True：中断那轮落的思考区在另一条路径上，
            # 本轮 collector 只有中断之后的片段。覆盖会让思考区整段消失。
            saved = storage.update_assistant_in_place(
                user_id,
                agent_id,
                session_id,
                regen_target_ai_id,
                ai_msg,
                extra=ai_extra,
                merge_thinking=True,
            )
        if not saved:
            yield {"type": "error", "content": "恢复失败：目标消息已被移除，请刷新后重试"}
            yield {"type": "done", "cancelled": False}
            return
    elif regenerate:
        # MCP 确认续跑原地覆盖目标助手消息；普通重生成插入兄弟版本并选中新版本
        saved = False
        if regen_target_ai_id is not None:
            if mcp_approved_pending_id:
                # 同 resume：MCP 确认续跑也跨越两次 collector
                saved = storage.update_assistant_in_place(
                    user_id,
                    agent_id,
                    session_id,
                    regen_target_ai_id,
                    ai_msg,
                    extra=ai_extra,
                    merge_thinking=True,
                )
            else:
                saved = storage.insert_assistant_version(
                    user_id, agent_id, session_id, regen_target_ai_id, ai_msg, extra=ai_extra
                )
        if not saved:
            yield {"type": "error", "content": "重新生成失败：目标回复已被移除，请刷新后重试"}
            yield {"type": "done", "cancelled": False}
            return
    else:
        storage.append_messages(
            user_id, agent_id, session_id, [ai_msg], extra_message_data=[ai_extra]
        )

    # 归档会话记忆（原话切片 → Milvus），按配置同步或后台线程
    schedule_archive_session_memory(user_id, agent_id, session_id)
    # U7：提炼跨会话长期记忆。与归档是两条独立的水位线：
    # archive 按"离开 compact 窗口"触发（苛刻），curation 按"攒够未提炼轮数"（宽松）。
    if llm_config:
        from app.chat.memory_curation import schedule_curate_session_memories

        schedule_curate_session_memories(user_id, agent_id, session_id, llm_config)

    yield {"type": "done", "cancelled": False}


async def iter_chat_stream_events(
    ua: UserAgent,
    user_text: str,
    user_id: int,
    agent_id: int,
    session_id: str,
    *,
    use_knowledge_retrieval: bool = True,
    use_web_search: bool = False,
    attachment_ids: list[str] | None = None,
    regenerate: bool = False,
    target_message_id: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
    mcp_approved_pending_id: str | None = None,
    job_id: str | None = None,
    resume_decision: str | None = None,
    resume_thread_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    异步对话：产出与 SSE 中 `data: {...}` 相同结构的 dict 事件（供直连 SSE 与后台 Job 复用）。
    最后依次产出 trace（若有）、写入存储后产出 done。

    本函数只做**编排**：按 ①→⑮ 的顺序调用上面的零件。改造前它是 757 行的一个函数，
    改对话行为时无法判断「改的这段属于哪个阶段」，因此按事件源拆开。

    :param ua: 智能体
    :param user_text: 用户文本（regenerate 时仅作校验用，模型输入以存储中目标用户消息为准）
    :param user_id: 用户 ID
    :param agent_id: 智能体 ID
    :param session_id: 会话 ID
    :param use_knowledge_retrieval: 是否启用知识库检索
    :param use_web_search: 是否启用联网搜索工具（与知识库检索互斥）
    :param regenerate: 为 True 时不追加用户消息，在目标助手消息的父用户消息下生成新版本
    :param target_message_id: regenerate 时要重写的助手消息 ID；缺省为当前路径末尾助手消息
    :param cancel_check: 若返回 True 则协作停止生成（如同步读 Redis 取消标记）
    :return: 异步迭代器
    """
    attachment_ids = attachment_ids or []
    # 节流后的取消探测：chunk 级调用不再每块都打 Redis 读取消标记，节流到 0.25s 一次
    cancel_probe = _throttled_cancel_check(cancel_check)

    # 清空 RAG 上下文, 重置工具调用守卫
    get_last_rag_context(clear=True)
    reset_tool_call_guards()
    set_turn_tool_policy(use_knowledge_retrieval=use_knowledge_retrieval, use_web_search=use_web_search)
    set_approved_mcp_pending_id(mcp_approved_pending_id)

    # ① resume 入参校验
    # 重新生成的目标助手消息 id：显式传 target_message_id 时按它定位（MCP 续跑原地覆盖它）；
    # 未指定时回退为当前路径末尾的助手消息。
    is_resume = bool(resume_decision)
    if is_resume:
        # 函数层再守一道：schema 已经校验过，但内部调用（脚本、测试、未来的其他入口）
        # 可能绕过 schema。非法值一路传到 HITL middleware 会抛 ValueError，
        # 并在那个 thread 上留下无法恢复的状态——比在入口拒绝糟糕得多。
        if str(resume_decision).strip().lower() not in ("approve", "reject"):
            yield {"type": "error", "content": "审批决定无效：只能是 approve 或 reject"}
            yield {"type": "done", "cancelled": False}
            return
        resume_decision = str(resume_decision).strip().lower()
        if not target_message_id:
            yield {"type": "error", "content": "恢复失败：缺少目标消息 id，请刷新后重试"}
            yield {"type": "done", "cancelled": False}
            return

    # ② 三路输入归一（普通提问 / regenerate / resume）
    inputs = _resolve_turn_inputs(
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        user_text=user_text,
        is_resume=is_resume,
        regenerate=regenerate,
        target_message_id=target_message_id,
        resume_thread_id=resume_thread_id,
    )
    if inputs.error:
        yield {"type": "error", "content": inputs.error}
        yield {"type": "done", "cancelled": False}
        return
    messages = inputs.messages
    regen_target_ai_id = inputs.regen_target_ai_id
    regen_path_ids = inputs.regen_path_ids

    # ③ 会话附件清单
    session_attachment_hint = format_attachment_hint(user_id, agent_id, session_id)

    # ④ 三路并行预取：选档知识库 / 会话记忆 / 工作区
    # 工作区与 KB/记忆预取并行：容器启动要 1–2s，串行会把首字延迟白白拉长。
    (
        retrieval_filter,
        kb_preselect_meta,
        memory_inject_prefetched,
        workspace,
    ) = await _prefetch_turn_side_effects(
        ua,
        messages,
        inputs.preselect_query,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        regenerate=regenerate,
        use_knowledge_retrieval=use_knowledge_retrieval,
        job_id=job_id,
    )

    # ⑤ 审批中断的运行时（只有带 job_id 的路径才有）
    thread_id = resume_thread_id or job_id
    checkpointer, interrupt_on = await _resolve_checkpointer(thread_id=thread_id)

    # ⑥ 加载该智能体已启用的 MCP 工具（单服务失败仅记录，不中断对话）；
    # 共享（非属主）会话跳过加载，避免属主凭据被共享用户对话驱动。
    if _mcp_tools_allowed_for(ua, user_id):
        mcp_tools, mcp_errors = await load_agent_mcp_tools(agent_id, user_id=user_id, session_id=session_id)
    else:
        mcp_tools, mcp_errors = [], []

    # ⑦ 构建模型和智能体
    agent, _model = build_model_and_agent(
        ua,
        user_id,
        agent_id,
        session_id,
        knowledge_base_document_filter=retrieval_filter if use_knowledge_retrieval else None,
        extra_tools=mcp_tools,
        use_knowledge_retrieval=use_knowledge_retrieval,
        use_web_search=use_web_search,
        workspace=workspace,
        checkpointer=checkpointer,
        interrupt_on=interrupt_on,
    )

    # ⑧ resume 前必须确认「断点还在」
    pending_action_count, resume_error = await _verify_resume_snapshot(
        agent, thread_id=thread_id, is_resume=is_resume, checkpointer=checkpointer
    )
    if resume_error:
        yield {"type": "error", "content": resume_error}
        yield {"type": "done", "cancelled": False}
        return

    # ⑨ 思考区事件总线：注册代理 → 写入 MCP 加载结果
    collector = _ThinkingCollector()
    collector.bind_rag_step_queue()
    if mcp_tools:
        collector.add_step("🧩", "MCP 工具已加载", f"{len(mcp_tools)} 个工具")
    for err in mcp_errors:
        collector.add_step("⚠️", f"MCP 服务「{err['name']}」不可用", err["error"][:120])

    if not regenerate and not is_resume:
        human_content = build_storable_human_content(
            user_text,
            attachment_ids,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            supports_vision=bool(getattr(ua, "supports_vision", False)),
        )
        human_msg = HumanMessage(content=human_content)
        messages.append(human_msg)
        # ★ 生成完成前就持久化用户消息，刷新后仍可从历史会话看到提问（助手在结束时再写入）
        await asyncio.to_thread(storage.append_messages, user_id, agent_id, session_id, [human_msg])

    # 当前上下文各消息对应的存储行 id（与 messages 等长对齐），供压缩/记忆按 turn_key 定位
    if is_resume:
        # 图从 checkpoint 恢复，历史由断点状态提供；这里只需保持行 id 对齐逻辑不炸
        path_ids = None
    elif regenerate:
        path_ids: list[int] | None = regen_path_ids
    else:
        path_records = storage.get_session_messages(user_id, agent_id, session_id)
        path_ids = [int(r.get("message_id") or 0) for r in path_records]
        if len(path_ids) != len(messages):
            path_ids = None

    memory_query = (user_text or "").strip()
    if regenerate and not is_resume:
        memory_query = msg_content_to_str(messages[-1].content).strip()

    # 先 flush 已排队步骤（MCP 加载结果等），随后的两阶段读图事件直接产出，
    # 保证思考区顺序与实时性（队列事件要到生成阶段才被消费）
    for event in collector.drain():
        yield event

    # ⑩ 图片预读：视觉模型先出文字描述
    outcome = _StreamOutcome()
    last_human_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human_msg is not None and _human_content_has_image_ref(last_human_msg.content):
        async for event in _run_vision_prepass(
            ua,
            messages,
            last_human_msg,
            collector,
            outcome,
            cancel_probe,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            use_knowledge_retrieval=use_knowledge_retrieval,
            use_web_search=use_web_search,
            has_mcp_tools=bool(mcp_tools),
        ):
            yield event
        if outcome.cancelled:
            get_last_rag_context(clear=True)
            get_pending_mcp_confirmations(clear=True)
            set_rag_step_queue(None)
            yield {"type": "cancelled"}
            yield {"type": "done", "cancelled": True}
            return

    # ⑪ 组装真正发给模型的消息（滑动窗口等）
    to_invoke = await asyncio.to_thread(
        _prepare_to_invoke_messages,
        messages,
        ua,
        user_id,
        agent_id,
        session_id,
        memory_query,
        use_knowledge_retrieval=use_knowledge_retrieval,
        use_web_search=use_web_search,
        document_filter=retrieval_filter if use_knowledge_retrieval else None,
        session_attachment_hint=session_attachment_hint,
        mcp_approval_note=_mcp_approval_note(
            mcp_approved_pending_id, user_id=user_id, agent_id=agent_id, session_id=session_id
        ),
        path_ids=path_ids,
        image_caption=collector.image_caption,
        memory_inject_prefetched=memory_inject_prefetched,
        skip_memory_retrieve=True,
    )

    # ⑫ 最后一次取消检查
    # 用户在这些阻塞阶段点击"停止"，cancel_probe 可能还没被及时检查到。
    # 现在所有预处理都完了、即将启动 Agent 正式生成之前，再查一次：
    # 如果用户已经点了停止，不启动 Agent 直接返回取消，省了白白消耗 LLM token。
    if cancel_probe is not None and await cancel_probe():
        get_last_rag_context(clear=True)  # 清理 RAG 上下文缓存
        get_pending_mcp_confirmations(clear=True)  # 清理待确认的 MCP 审批
        set_rag_step_queue(None)  # 注销思考区事件总线
        yield {"type": "cancelled"}
        yield {"type": "done", "cancelled": True}
        return

    # ⑬ 启动生成 + 取消看门狗
    # updates 流有两个用途：抓审批中断（需 checkpointer）、抓待办清单（规划）。
    # 规划不依赖 checkpointer，所以开关打开时即使没有 thread_id 也要订阅 updates。
    use_updates = checkpointer is not None or _todo_enabled()
    # resume 时输入必须是 Command(resume=...)，图才会从中断点继续而不是从头跑。
    # ⚠️ decisions 的每一项是 `{"type": "approve"}` 这样的 dict，不是裸字符串——
    # middleware 内部按 `decision["type"]` 取值，传字符串会在恢复时炸掉。
    if resume_decision:
        from langgraph.types import Command

        # 每个待确认工具各配一个决定；数量不足 middleware 会抛
        # `Number of human decisions (1) does not match ...`
        stream_input: Any = Command(
            resume={
                "decisions": [
                    {"type": resume_decision} for _ in range(max(1, pending_action_count))
                ]
            }
        )
    else:
        stream_input = {"messages": to_invoke}

    agent_task = asyncio.create_task(
        _run_agent_stream(
            agent,
            stream_input,
            collector,
            outcome,
            cancel_probe,
            use_updates=use_updates,
            thread_id=thread_id,
        )
    )
    watchdog_task = asyncio.create_task(_cancel_watchdog(agent_task, cancel_probe))

    try:
        while True:
            event = await collector.queue.get()
            if event is None:
                break
            yield event  # ★★★★★ 唯一出口
    except GeneratorExit:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        set_rag_step_queue(None)
        watchdog_task.cancel()
        if not agent_task.done():
            agent_task.cancel()
        # 工作区清理放这里而不是外面包一层大 try：这一层已经覆盖了「正常结束」与
        # 「客户端断线（GeneratorExit）」两条路径，而后续收尾（trace/sources/落库）
        # 不再调用任何工具，容器可以提前销毁。
        #
        # 用同步 cleanup 而非 `await asyncio.to_thread(...)`：async generator 在
        # `aclose()` 期间如果 finally 里出现 await，Python 会抛
        # "async generator ignored GeneratorExit"。代价是阻塞 200–500ms，
        # 但只在流结束时发生一次。
        if workspace is not None:
            try:
                workspace.cleanup()
            except Exception:  # noqa: BLE001
                logger.warning("工作区清理失败（不影响结果）", exc_info=True)

    # ⑭ 取消收尾
    if outcome.cancelled:
        get_last_rag_context(clear=True)
        get_pending_mcp_confirmations(clear=True)
        yield {"type": "cancelled"}
        yield {"type": "done", "cancelled": True}
        return

    # ⑮ 中断收尾 / 正常收尾
    if outcome.interrupted:
        async for event in _finish_interrupted(
            collector,
            outcome,
            (user_id, agent_id, session_id),
            thread_id=thread_id,
            # resume 后又中断 → 原地覆盖那条待确认消息，不再堆新卡片
            is_resume=is_resume,
            regen_target_ai_id=regen_target_ai_id,
        ):
            yield event
        return

    # curation 需要 LLM 配置。build_model_and_agent 内部算过一份但没返回，
    # 这里单独取（纯读配置 + 解密，无网络开销）。
    memory_llm_config = _sub_llm_config_from_ua(ua)

    async for event in _finish_completed(
        collector,
        outcome,
        messages,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        is_resume=is_resume,
        regenerate=regenerate,
        regen_target_ai_id=regen_target_ai_id,
        mcp_approved_pending_id=mcp_approved_pending_id,
        kb_preselect_meta=kb_preselect_meta,
        llm_config=memory_llm_config,
    ):
        yield event


async def chat_with_agent_stream(
    ua: UserAgent,
    user_text: str,
    user_id: int,
    agent_id: int,
    session_id: str,
    *,
    use_knowledge_retrieval: bool = True,
    use_web_search: bool = False,
    attachment_ids: list[str] | None = None,
    regenerate: bool = False,
    target_message_id: int | None = None,
    mcp_approved_pending_id: str | None = None,
) -> AsyncIterator[str]:
    """
    异步对话（SSE 字符串片段）
    :param ua: 智能体
    :param user_text: 用户文本
    :param user_id: 用户 ID
    :param agent_id: 智能体 ID
    :param session_id: 会话 ID
    :param use_knowledge_retrieval: 是否启用知识库检索
    :param use_web_search: 是否启用联网搜索工具（与知识库检索互斥）
    :param regenerate: 是否重新生成助手回复
    :param target_message_id: regenerate 时要重写的助手消息 ID；缺省为当前路径末尾助手消息
    :return: 异步迭代器
    """
    # 调用智能体异步对话函数，流式返回响应，【入口】
    async for ev in iter_chat_stream_events(
        ua,
        user_text,
        user_id,
        agent_id,
        session_id,
        use_knowledge_retrieval=use_knowledge_retrieval,
        use_web_search=use_web_search,
        attachment_ids=attachment_ids,
        regenerate=regenerate,
        target_message_id=target_message_id,
        mcp_approved_pending_id=mcp_approved_pending_id,
    ):
        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"