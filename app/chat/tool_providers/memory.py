"""会话记忆工具 provider（search_session_memory）。"""

from __future__ import annotations

from typing import Any

from app.chat.tool_providers.base import ToolBuildContext
from app.settings import settings


class MemoryToolProvider:
    """当前会话的向量记忆检索。"""

    name = "memory"
    order = 40

    def enabled(self, ctx: ToolBuildContext) -> bool:
        return bool(getattr(settings, "CHAT_USE_SESSION_MEMORY", True))

    def build(self, ctx: ToolBuildContext) -> list[Any]:
        from app.chat.memory_tool import make_search_session_memory_tool

        return [
            make_search_session_memory_tool(
                ctx.user_id, ctx.agent_id, ctx.session_id, ctx.llm_config
            )
        ]

    def prompt_fragments(self, ctx: ToolBuildContext) -> list[str]:
        # 记忆工具的说明写在 turn context block 里（"若仍不足可再调用 search_session_memory"），
        # 因为它是"已经在 human 里注入了摘录"这个事实的补充，属每轮上下文。
        return []


PROVIDER = MemoryToolProvider()
