"""记忆工具 provider：会话级（search_session_memory）+ 跨会话长期记忆（search_long_term_memory）。

两个工具**并列**而不是合并：
- `search_session_memory`：本会话里被 compact 挤出窗口的**原话**切片
- `search_long_term_memory`：以往所有会话提炼出的**稳定事实**

语义不同，合并会让模型分不清该查哪个；配额也各自独立（各限一次/轮），
共用一个会让模型被迫二选一，而正确答案往往是两边都要看。

两个开关也彼此独立：会话记忆靠 `CHAT_USE_SESSION_MEMORY`，
长期记忆靠 `CHAT_MEMORY_CURATION_ENABLED` + `CHAT_MEMORY_USER_SCOPE_ENABLED`。
任一个关掉另一个照常工作——这是刻意的：会话记忆的触发条件（compact）比
长期记忆苛刻得多，绑在一起会让长期记忆被它拖累成不可用。
"""

from __future__ import annotations

from typing import Any

from app.chat.tool_providers.base import ToolBuildContext
from app.settings import settings


def _session_memory_on() -> bool:
    return bool(getattr(settings, "CHAT_USE_SESSION_MEMORY", True))


def _long_term_memory_on() -> bool:
    from app.chat.memory_curation import user_scope_enabled

    return user_scope_enabled()


class MemoryToolProvider:
    """记忆检索（会话级 + 跨会话）。"""

    name = "memory"
    order = 40

    def enabled(self, ctx: ToolBuildContext) -> bool:
        # 任一开关打开就该装（build 里再按各自开关决定装哪个）
        return _session_memory_on() or _long_term_memory_on()

    def build(self, ctx: ToolBuildContext) -> list[Any]:
        tools: list[Any] = []
        if _session_memory_on():
            from app.chat.memory_tool import make_search_session_memory_tool

            tools.append(
                make_search_session_memory_tool(
                    ctx.user_id, ctx.agent_id, ctx.session_id, ctx.llm_config
                )
            )
        if _long_term_memory_on():
            from app.chat.memory_tool import make_search_long_term_memory_tool

            tools.append(make_search_long_term_memory_tool(ctx.user_id, ctx.agent_id))
        return tools

    def prompt_fragments(self, ctx: ToolBuildContext) -> list[str]:
        # 记忆工具的说明写在 turn context block 里（"若仍不足可再调用 ..."），
        # 因为它是"已经在 human 里注入了摘录"这个事实的补充，属每轮上下文。
        return []


PROVIDER = MemoryToolProvider()
