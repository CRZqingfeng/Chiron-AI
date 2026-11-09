"""每轮 agent 运行时状态（原 `app/chat/tools.py`）。

为什么在 `app/core` 而不在 `app/chat`：这份状态被三方共用——`app/chat`（工具实现）、
`app/kb`（检索流程要 emit RAG 步骤）。
放在 `app/chat` 下会让知识库反向依赖对话域，故下沉为中性基础设施。

知识库/记忆/联网检索的具体工具由各自模块动态绑定，这里只管状态与配额。

并发安全：全部状态挂在按请求创建的 _RequestState 上，经 ContextVar 分发。
asyncio.create_task / asyncio.to_thread / LangChain run_in_executor 均会复制当前
contextvars 上下文（携带同一状态对象引用），同步路径整体运行于单线程内，
因此并发请求互不可见，杜绝跨请求（跨用户）RAG 上下文与检索步骤串扰。
"""

from __future__ import annotations

import asyncio
import contextvars
import re
from typing import Any, Optional


class _RequestState:
    """单次对话请求内的工具共享状态（每请求一个实例，随上下文隔离）。"""

    __slots__ = (
        "last_rag_context",
        "knowledge_calls",
        "memory_calls",
        "long_term_memory_calls",
        "image_kb_calls",
        "web_search_calls",
        "web_image_search_calls",
        "fetch_url_calls",
        "rag_step_queue",
        "rag_step_loop",
        "allow_knowledge_retrieval",
        "allow_web_search",
        "session_scope",
    )

    def __init__(self) -> None:
        self.last_rag_context: dict | None = None
        self.knowledge_calls = 0
        self.memory_calls = 0
        self.long_term_memory_calls = 0
        self.image_kb_calls = 0
        self.web_search_calls = 0
        self.web_image_search_calls = 0
        self.fetch_url_calls = 0
        self.rag_step_queue: Any = None
        self.rag_step_loop: asyncio.AbstractEventLoop | None = None
        self.allow_knowledge_retrieval = True
        self.allow_web_search = False
        # 「用户×智能体×会话」标识。审批中间件靠它判断「本会话是否已授权跑测试」——
        # 中间件只能拿到 state/runtime，而 `thread_id` 是 job_id 不是会话，拿不到这个。
        self.session_scope: str | None = None


_REQUEST_STATE: contextvars.ContextVar[_RequestState | None] = contextvars.ContextVar(
    "chiron_agent_tool_request_state", default=None
)


def _state() -> _RequestState:
    state = _REQUEST_STATE.get()
    if state is None:
        state = _RequestState()
        _REQUEST_STATE.set(state)
    return state


def _set_last_rag_context(context: dict) -> None:
    _state().last_rag_context = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    state = _state()
    context = state.last_rag_context
    if clear:
        state.last_rag_context = None
    return context


def reset_tool_call_guards() -> None:
    state = _state()
    state.knowledge_calls = 0
    state.memory_calls = 0
    state.long_term_memory_calls = 0
    state.image_kb_calls = 0
    state.web_search_calls = 0
    state.web_image_search_calls = 0
    state.fetch_url_calls = 0


def make_session_scope(user_id: int, agent_id: int, session_id: str) -> str:
    """把「用户×智能体×会话」压成一个可放进 Redis key 的稳定标识。

    三者都要：只用 session_id 不够——会话 id 由前端生成，不同用户之间可能撞名。
    """
    raw = f"u{int(user_id)}_a{int(agent_id)}_s{session_id or ''}"
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:120]


def set_session_scope(scope: str | None) -> None:
    """记录本轮属于哪个会话。

    为什么需要它：审批中间件（`app.chat.approval`）只能拿到 state / runtime，
    而 `thread_id` 是 **job_id 不是 session_id**（`agent_service` 里
    `thread_id = resume_thread_id or job_id`）。没有这个标识，
    「本会话已授权跑测试」这类会话级判定就无处可查。
    """
    _state().session_scope = (scope or "").strip() or None


def get_session_scope() -> str | None:
    return _state().session_scope


def set_turn_tool_policy(*, use_knowledge_retrieval: bool, use_web_search: bool) -> None:
    """本轮检索开关（工具始终挂载，禁用时由工具函数返回 TOOL_DISABLED_THIS_TURN）。"""
    state = _state()
    state.allow_knowledge_retrieval = bool(use_knowledge_retrieval)
    state.allow_web_search = bool(use_web_search)


def is_knowledge_allowed_this_turn() -> bool:
    return bool(_state().allow_knowledge_retrieval)


def is_web_search_allowed_this_turn() -> bool:
    return bool(_state().allow_web_search)


def knowledge_disabled_this_turn_msg(tool_name: str) -> str:
    return (
        f"TOOL_DISABLED_THIS_TURN: {tool_name} is not enabled for this turn. "
        "Do not call knowledge-base retrieval tools; answer without them."
    )


def web_search_disabled_this_turn_msg() -> str:
    return (
        "TOOL_DISABLED_THIS_TURN: web_search is not enabled for this turn. "
        "Do not call web_search or fetch_url; answer without live web results."
    )


def fetch_url_disabled_this_turn_msg() -> str:
    return (
        "TOOL_DISABLED_THIS_TURN: fetch_url is not enabled for this turn. "
        "Do not call fetch_url or web_search; answer without live web results."
    )


def web_image_search_disabled_this_turn_msg() -> str:
    return (
        "TOOL_DISABLED_THIS_TURN: web_image_search is not enabled for this turn. "
        "Do not call web_image_search; answer without live web images."
    )


def try_acquire_knowledge_tool_slot() -> bool:
    """本轮知识库检索配额，成功占用返回 True。

    上限由 KNOWLEDGE_MAX_CALLS_PER_TURN 控制（默认 2）。此前硬编码为 1：
    模型第一次检索没捞到关键证据时无法追问第二个角度，多跳问题（先查 A、再据 A 查 B）必然断链。
    放开后由 recursion_limit 与工具内部去重兜底，不会无界循环。
    """
    from app.settings import settings

    state = _state()
    max_n = max(1, int(getattr(settings, "KNOWLEDGE_MAX_CALLS_PER_TURN", 2) or 1))
    if state.knowledge_calls >= max_n:
        return False
    state.knowledge_calls += 1
    return True


def try_acquire_memory_tool_slot() -> bool:
    """同一轮对话仅允许一次会话记忆检索；成功占用返回 True。"""
    state = _state()
    if state.memory_calls >= 1:
        return False
    state.memory_calls += 1
    return True


def try_acquire_long_term_memory_tool_slot() -> bool:
    """同一轮对话仅允许一次**长期记忆**检索；成功占用返回 True。

    与会话记忆配额分开计：两者语义不同（"本会话早期说过什么" vs "以往会话的稳定事实"），
    共用一个计数会让模型被迫二选一，而正确答案往往是两边都要看。
    """
    state = _state()
    if state.long_term_memory_calls >= 1:
        return False
    state.long_term_memory_calls += 1
    return True


def try_acquire_image_kb_tool_slot(max_n: int) -> bool:
    """以图知识库检索配额：本轮成功次数不超过 max_n，成功占用返回 True。

    为什么要传 max_n 而不是自己查附件表：上限等于「本会话图片类附件数」，
    那个计数在对话域的附件表里。如果这里自己去 import `app.chat.attachment_service`，
    core 就反向依赖了 chat。计数交给调用方（`app/kb/image_search_tool`），
    本函数只做纯粹的配额判断。
    U8b 把图源抽象出来之后，调用方可以换成别的计数来源，这里无需改动。
    """
    if max_n <= 0:
        return False
    state = _state()
    if state.image_kb_calls >= max_n:
        return False
    state.image_kb_calls += 1
    return True


def try_acquire_web_search_tool_slot() -> bool:
    """同一轮对话限制联网搜索次数（防 ReAct 循环刷限流）；成功占用返回 True。"""
    from app.settings import settings

    state = _state()
    max_n = max(1, int(getattr(settings, "WEB_SEARCH_MAX_CALLS_PER_TURN", 2)))
    if state.web_search_calls >= max_n:
        return False
    state.web_search_calls += 1
    return True


def try_acquire_fetch_url_tool_slot() -> bool:
    """同一轮对话限制 fetch_url 次数；成功占用返回 True。"""
    from app.settings import settings

    state = _state()
    max_n = max(1, int(getattr(settings, "WEB_SEARCH_FETCH_MAX_CALLS_PER_TURN", 3)))
    if state.fetch_url_calls >= max_n:
        return False
    state.fetch_url_calls += 1
    return True


def try_acquire_web_image_search_tool_slot() -> bool:
    """同一轮对话限制文字搜图次数；成功占用返回 True。"""
    from app.settings import settings

    state = _state()
    max_n = max(1, int(getattr(settings, "WEB_IMAGE_SEARCH_MAX_CALLS_PER_TURN", 2)))
    if state.web_image_search_calls >= max_n:
        return False
    state.web_image_search_calls += 1
    return True


def set_rag_step_queue(queue: Any, *, sync: bool = False) -> None:
    state = _state()
    state.rag_step_queue = queue
    if queue is None:
        state.rag_step_loop = None
    elif sync:
        state.rag_step_loop = None
    else:
        try:
            state.rag_step_loop = asyncio.get_running_loop()
        except RuntimeError:
            state.rag_step_loop = asyncio.get_event_loop()


def log_kb_tool_return_to_terminal(text: str, *, tool_label: str = "search_knowledge_base") -> None:
    """在终端打印知识库/联网工具返回给模型的字符串（受 DEBUG_AGENT_KB_PROMPT 控制）。"""
    from app.settings import settings

    if not getattr(settings, "DEBUG_AGENT_KB_PROMPT", False):
        return
    sep = "=" * 72
    print(
        f"\n{sep}\n[智能体工具] {tool_label} 工具输出（将注入对话上下文）:\n{sep}\n{text}\n{sep}\n",
        flush=True,
    )


def emit_rag_step(icon: str, label: str, detail: str = "") -> None:
    state = _state()
    queue = state.rag_step_queue
    if queue is None:
        return
    step = {"icon": icon, "label": label, "detail": detail}
    loop = state.rag_step_loop
    if loop is not None:
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, step)
        except Exception:
            pass
    else:
        try:
            queue.put_nowait(step)
        except Exception:
            pass
