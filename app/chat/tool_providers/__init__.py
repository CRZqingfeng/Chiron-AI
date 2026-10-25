"""工具 provider 的有序注册表与装配入口。

新增一个工具：

1. 新建 `xxx.py`，实现 `ToolProvider`（`enabled` / `build` / `prompt_fragments`）
2. 在 `app/chat/tool_registry.py` 的 `TOOL_SPECS` 登记工具名
3. 在下面 `TOOL_PROVIDERS` 加一行

`agent_service.build_model_and_agent` 不需要任何改动。

## 顺序为什么重要

`order` 决定两件事，都影响模型行为：

- **工具在列表里的位置**：装配后还会 `_sort_tools` 按名字排序，所以最终顺序与
  provider 顺序无关；但 `build()` 内部的产出顺序会影响重名检测时的先到先得。
- **system prompt 片段顺序**：按 `order` 拼装，不排序。改造前的顺序是
  「人设 → 联网纪律 → KB 图片 → KB 作答」，所以 web(20) 必须早于 kb(30)。
  `tests/chat/test_tool_providers.py` 有逐字节对拍守卫这个顺序。
"""

from __future__ import annotations

from typing import Any

from app.chat.tool_providers import attachment, kb, mcp, memory, web, workspace
from app.chat.tool_providers.base import ToolBuildContext, ToolProvider

TOOL_PROVIDERS: tuple[ToolProvider, ...] = tuple(
    sorted(
        (
            attachment.PROVIDER,
            web.PROVIDER,
            kb.PROVIDER,
            memory.PROVIDER,
            workspace.PROVIDER,
            mcp.PROVIDER,
        ),
        key=lambda p: p.order,
    )
)


def build_tools(ctx: ToolBuildContext) -> tuple[list[Any], list[str]]:
    """按注册表装配工具与 system prompt 片段。

    :return: `(tools, prompt_fragments)`。片段已按 provider 顺序展开，
        调用方直接 `"\\n\\n".join` 到人设之后即可。
    """
    tools: list[Any] = []
    fragments: list[str] = []
    for provider in TOOL_PROVIDERS:
        if not provider.enabled(ctx):
            continue
        tools.extend(provider.build(ctx))
        fragments.extend(provider.prompt_fragments(ctx))
    return tools, fragments


__all__ = ["TOOL_PROVIDERS", "ToolBuildContext", "ToolProvider", "build_tools"]
