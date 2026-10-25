"""内置工具的注册表 —— 工具名的唯一来源。

## 为什么需要这个文件

改造前，「工具名」散落在四处，靠人工同步：

1. `agent_service.build_model_and_agent` 里的注册点
2. `mcp_client/service.py` 的 `_BUILTIN_TOOL_NAMES`（MCP 撞名检测用）
3. `agent_prompt.py` 的提示词（写死了 `web_search` / `search_knowledge_base` 等）
4. 前端 `useAgentConfigDiff.js` 的三份提示词副本

代价已经付过了：U4 新增 4 个 `workspace_*` 工具时只改了第 1 处，
第 2 处的最初实现漏了这 4 个名字。后果不是报错，而是**静默**——

    MCP server 若暴露一个叫 workspace_write_file 的工具
    → 通过撞名检测（不在白名单）
    → 与内置工具同名进入 create_agent
    → LangChain 静默保留其一，**不抛异常**（已实测）
    → 模型以为在写沙箱文件，实际可能走到 MCP 那侧，绕过容器隔离

更隐蔽的是：审批按工具名匹配（`interrupt_on`），所以 `workspace_run_command`
被顶替时审批照样触发 —— 但拦下的不是真正的沙箱执行。这比不设审批更难发现。

## 本文件解决什么

- `TOOL_SPECS` 是唯一清单；`BUILTIN_TOOL_NAMES` 由它派生，不可能再漏
- `assert_no_duplicates`：装配期发现重名直接抛错，不让静默覆盖发生
- `warn_unknown_tools`：工具名未登记时告警（不阻塞对话，因为 MCP 工具是动态的）

## 新增工具时

在 `TOOL_SPECS` 里加一行。忘加不会立刻炸，但 `warn_unknown_tools` 会打日志，
`tests/chat/test_tool_registry.py` 的覆盖性断言会红。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    """一条内置工具的登记信息。"""

    name: str
    """工具名（必须与 `@tool` 装饰后的 `.name` 完全一致）。"""

    category: str
    """分类：attachment / web / kb / memory / workspace。仅用于分组阅读与日志。"""

    owner: str
    """定义所在模块路径。排错时先看这里。"""

    builtin: bool = True
    """是否计入 MCP 撞名白名单。MCP 动态工具不登记，故不需要 False。"""


TOOL_SPECS: tuple[ToolSpec, ...] = (
    # 会话附件（三件套）
    ToolSpec("search_session_attachment", "attachment", "app/chat/attachment_tools.py"),
    ToolSpec("read_session_attachment", "attachment", "app/chat/attachment_tools.py"),
    ToolSpec("list_session_attachments_brief", "attachment", "app/chat/attachment_tools.py"),
    # 联网（三件套，受 WEB_SEARCH_ENABLED 控制）
    ToolSpec("web_search", "web", "app/chat/web_search_tool.py"),
    ToolSpec("fetch_url", "web", "app/chat/web_search_tool.py"),
    ToolSpec("web_image_search", "web", "app/chat/web_search_tool.py"),
    # 知识库
    ToolSpec("search_knowledge_base", "kb", "app/kb/search_tool.py"),
    ToolSpec("search_knowledge_by_image", "kb", "app/kb/image_search_tool.py"),
    # 会话记忆（受 CHAT_USE_SESSION_MEMORY 控制）
    ToolSpec("search_session_memory", "memory", "app/chat/memory_tool.py"),
    # 跨会话长期记忆（U7，受 CHAT_MEMORY_CURATION_ENABLED + USER_SCOPE_ENABLED 控制）
    ToolSpec("search_long_term_memory", "memory", "app/chat/memory_tool.py"),
    # 工作区（受 workspace 是否创建 + interrupt_on 控制）
    ToolSpec("workspace_list_dir", "workspace", "app/workspace/tools.py"),
    ToolSpec("workspace_read_file", "workspace", "app/workspace/tools.py"),
    ToolSpec("workspace_write_file", "workspace", "app/workspace/tools.py"),
    ToolSpec("workspace_run_command", "workspace", "app/workspace/tools.py"),
)


BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(s.name for s in TOOL_SPECS if s.builtin)
"""内置工具名集合。MCP 工具与之撞名时跳过（保留内置实现）。

替代原先硬编码在 `mcp_client/service.py` 里的同名集合 —— 那份漏了 4 个
`workspace_*` 工具，详见模块 docstring。
"""

RESERVED_TOOL_NAMES: frozenset[str] = frozenset({"write_todos"})
"""由**中间件**注入、不在 `TOOL_SPECS` 里的工具名（U9a）。

为什么不并进 `BUILTIN_TOOL_NAMES`：那不是 provider 造出来的工具，把它塞进
`TOOL_SPECS` 会让「工具清单」和「provider 装配」对不上。但 MCP 撞名过滤
（`mcp_client/service.py`）必须认得它——否则某个 MCP server 也提供
`write_todos` 时，会出现两个同名工具绑到同一个模型，谁生效取决于顺序。
"""


def _names_of(tools: Iterable[Any]) -> list[str]:
    return [str(getattr(t, "name", "") or "") for t in tools]


def find_duplicates(tools: Iterable[Any]) -> dict[str, int]:
    """返回 `{工具名: 出现次数}`，只含出现超过一次的。空名忽略。"""
    counts: dict[str, int] = {}
    for name in _names_of(tools):
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    return {n: c for n, c in counts.items() if c > 1}


def assert_no_duplicates(tools: Iterable[Any]) -> None:
    """装配期检查：工具重名直接抛错。

    为什么不"跳过重复项了事"：静默覆盖是安全问题（见模块 docstring 的风险链），
    而抛错是可见的、可修的。这里宁可让配置错误暴露，也不让模型拿到一个
    它以为安全、实际被顶替的工具。
    """
    dups = find_duplicates(tools)
    if not dups:
        return
    detail = "、".join(f"{name}×{count}" for name, count in sorted(dups.items()))
    raise ValueError(
        f"工具名重复：{detail}。内置工具见 app/chat/tool_registry.py 的 TOOL_SPECS；"
        "若来自 MCP，请改名或移除该 server（重名会导致内置实现被静默覆盖）。"
    )


def warn_unknown_tools(tools: Iterable[Any], *, exempt: Sequence[str] = ()) -> list[str]:
    """工具名未登记时告警，返回未登记清单。

    不抛异常：MCP 工具是动态的、本来就不在 `TOOL_SPECS` 里（由 `exempt` 排除）。
    真正该修的遗漏（新增内置工具忘了登记）会在日志里显形，并由
    `tests/chat/test_tool_registry.py` 的覆盖性断言拦住。
    """
    known = BUILTIN_TOOL_NAMES | set(exempt)
    unknown = sorted({n for n in _names_of(tools) if n and n not in known})
    if unknown:
        logger.warning(
            "以下工具未在 tool_registry.TOOL_SPECS 登记（MCP 动态工具请确认已在 exempt 中）：%s",
            unknown,
        )
    return unknown
