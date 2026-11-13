"""工具网关 provider：产出 `search_tools` 与对应的提示词片段。

## 为什么是一个 provider 而不是塞进 mcp provider

`mcp` provider 的职责是「把已加载的工具纳入统一装配」，它对工具数量不感兴趣；
网关的职责是「工具太多时，提供一个发现入口」。两者条件不同：MCP 工具 3 个时
前者要工作、后者不该出现。混在一起会让 `enabled` 变成两件事的或。

## 为什么提示词片段写在这里

和 `web` / `kb` 的做法一致：「这个工具怎么用」和「这个工具长什么样」写在一个文件里，
改一半的情况不会发生。而且片段只在网关启用时才拼进 system prompt，
未超阈值时模型收到的提示词**逐字不变**。
"""

from __future__ import annotations

from typing import Any

from app.chat.tool_gateway import GATEWAY_PROMPT_FRAGMENT, gateway_active, make_search_tools_tool
from app.chat.tool_providers.base import ToolBuildContext


class ToolGatewayProvider:
    """按需工具发现（MCP 工具超过阈值时才出现）。"""

    name = "tool_gateway"
    # 排最后：它是"补充说明"，应该跟在具体工具纪律之后
    order = 70

    def enabled(self, ctx: ToolBuildContext) -> bool:
        return gateway_active(len(ctx.extra_tools or []))

    def build(self, ctx: ToolBuildContext) -> list[Any]:
        return [make_search_tools_tool(list(ctx.extra_tools or []))]

    def prompt_fragments(self, ctx: ToolBuildContext) -> list[str]:
        return [GATEWAY_PROMPT_FRAGMENT]


PROVIDER = ToolGatewayProvider()
