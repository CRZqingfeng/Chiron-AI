"""会话附件工具 provider（三件套：检索 / 读取 / 列表）。"""

from __future__ import annotations

from typing import Any

from app.chat.tool_providers.base import ToolBuildContext


class AttachmentToolProvider:
    """本轮会话上传的附件；不需要全局开关，也不需要提示词纪律。"""

    name = "attachment"
    order = 10

    def enabled(self, ctx: ToolBuildContext) -> bool:
        return True

    def build(self, ctx: ToolBuildContext) -> list[Any]:
        from app.chat.attachment_tools import make_session_attachment_tools

        return make_session_attachment_tools(ctx.user_id, ctx.agent_id, ctx.session_id)

    def prompt_fragments(self, ctx: ToolBuildContext) -> list[str]:
        # 附件的使用说明随附图一起写在 human message 里（见 attachment_service），
        # 不适合放进 system prompt：它是每轮变化的上下文，不是长期纪律。
        return []


PROVIDER = AttachmentToolProvider()
