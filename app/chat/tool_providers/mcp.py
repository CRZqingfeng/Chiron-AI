"""MCP 工具 provider。

MCP 工具**不是在这里加载的**：`load_agent_mcp_tools` 是异步的，还要把连接失败
写进思考区步骤，逻辑在 `iter_chat_stream_events` 里（与知识库预取并行）。
所以本 provider 只做一件事：把已加载的列表纳入统一装配流程，让它走同一套
重名检查与排序，而不是在 `build_model_and_agent` 里单独 `extend` 一下。

审批实现（`wrap_mcp_tool_with_confirmation`）保持原样 ——
把它切到 HITL 是 U5c 的事，与本单元正交。
"""

from __future__ import annotations

from typing import Any

from app.chat.tool_providers.base import ToolBuildContext


class McpToolProvider:
    """外部 MCP 工具（动态、数量不定）。"""

    name = "mcp"
    order = 60

    def enabled(self, ctx: ToolBuildContext) -> bool:
        return bool(ctx.extra_tools)

    def build(self, ctx: ToolBuildContext) -> list[Any]:
        return list(ctx.extra_tools)

    def prompt_fragments(self, ctx: ToolBuildContext) -> list[str]:
        # MCP 工具的用法由各自的 schema/docstring 描述，服务方自己负责。
        return []


PROVIDER = McpToolProvider()
