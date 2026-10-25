"""工作区工具 provider（列目录 / 读 / 写 / 执行命令）。

这里有一条**安全约束**必须保留：`workspace_run_command` 只在「审批中断确实生效」
时才注册。它的判据是 `interrupt_on` 里有没有这个工具——不是另配一个开关。

理由：命令执行和它的审批保护必须同生同死。任何"有执行能力但没有审批"的中间状态
都是安全事故，而不靠两处配置保持一致来避免，靠的是「一个来源推导」（U5a-min 的设计）。
"""

from __future__ import annotations

from typing import Any

from app.chat.tool_providers.base import ToolBuildContext


class WorkspaceToolProvider:
    """沙箱工作区。`ctx.workspace` 为 None 时整体不启用。"""

    name = "workspace"
    order = 50

    def enabled(self, ctx: ToolBuildContext) -> bool:
        return ctx.workspace is not None

    def build(self, ctx: ToolBuildContext) -> list[Any]:
        from app.workspace.tools import make_workspace_tools

        interrupt_on = ctx.interrupt_on or {}
        allow_exec = bool(interrupt_on.get("workspace_run_command"))
        return make_workspace_tools(ctx.workspace, include_exec=allow_exec)

    def prompt_fragments(self, ctx: ToolBuildContext) -> list[str]:
        # 工作区工具目前靠 @tool 的 docstring 自解释，没有额外纪律。
        # 若将来要教模型「先 ls 再读、改动前先备份」，片段加在这里。
        return []


PROVIDER = WorkspaceToolProvider()
