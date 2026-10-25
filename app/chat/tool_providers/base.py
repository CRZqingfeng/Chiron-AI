"""工具提供者（ToolProvider）的协议与构建上下文。

## 为什么要有这一层

改造前，`build_model_and_agent` 里是 9 段硬编码的 `tools.append(...)`，
每段的启用条件（`if settings.X`）、构造参数、以及对应的提示词片段分散在三处。
加一个工具要改 7 个地方（见 `docs/dev/U6-思路.md` §2.5）。

Provider 化之后，加一个工具 = 新建一个 provider 文件 + 在 `__init__.py` 的
`TOOL_PROVIDERS` 加一行。`agent_service` 不再需要知道任何具体工具。

## ToolBuildContext 与「显式注入 TurnContext」的区别

方案原稿要求用 `TurnContext` 显式注入替代 `app/chat/tools.py` 的 ContextVar。
**这一步没有做**，原因是 `TurnContext` 会进 graph state、进而进 checkpoint，而
`workspace`（docker 容器句柄）与 `rag_step_queue`（`asyncio.Queue`）都不可序列化。

`ToolBuildContext` 是**构建期的普通参数对象**：只在 `build_model_and_agent`
调用栈里活几毫秒，不进 state、不进 checkpoint。工具仍然靠闭包捕获它需要的值，
ContextVar 继续承担「运行期共享状态」（配额、RAG 上下文、步骤队列）。

两者的分工：
- `ToolBuildContext`：决定「构建哪些工具、用什么参数构建」
- ContextVar（`app/chat/tools.py`）：决定「工具运行时能调几次、往哪推步骤」
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolBuildContext:
    """构建工具所需的全部上下文（构建期有效，不跨请求、不进 checkpoint）。"""

    ua: Any
    """用户智能体记录。工具可能需要 `supports_vision` 等配置。"""

    user_id: int
    agent_id: int
    session_id: str

    kb_scope: Any = None
    """知识库检索的作用域（`kb_scope_for` 的返回值）。"""

    llm_config: dict[str, Any] = field(default_factory=dict)
    """子模型配置（知识库/记忆工具内部会用它做改写、打分）。"""

    workspace: Any = None
    """已打开的工作区实例；None 表示不注册工作区工具。

    ⚠️ 不可序列化（含 docker 容器句柄）——这正是它不能进 graph state 的原因。
    """

    interrupt_on: dict[str, Any] | None = None
    """需要人工审批的工具名映射；工作区据此决定是否注册 `run_command`。"""

    extra_tools: list[Any] = field(default_factory=list)
    """外部加载的工具（目前只有 MCP）。由 MCP provider 原样返回。"""

    knowledge_base_document_filter: list[str] | None = None
    """本回合限定检索的 file_key 列表；None 表示整库检索。"""

    use_knowledge_retrieval: bool = True
    use_web_search: bool = False


@runtime_checkable
class ToolProvider(Protocol):
    """一类工具的提供者。"""

    name: str
    """provider 标识（用于日志与测试定位，不必等于工具名）。"""

    order: int
    """装配顺序。决定工具在列表里的相对位置，进而影响 system prompt 片段顺序。

    现有 prompt 顺序是「人设 → 联网纪律 → KB 图片 → KB 作答」，所以 web 必须
    排在 kb 之前。这个顺序一旦变了，模型收到的提示词就变了——`tests` 里
    `test_prompt_matches_baseline` 会逐字节对拍拦住。
    """

    def enabled(self, ctx: ToolBuildContext) -> bool:
        """是否启用本 provider。返回 False 时 `build` 与 `prompt_fragments` 都不会被调用。"""
        ...

    def build(self, ctx: ToolBuildContext) -> list[Any]:
        """构造工具列表。"""
        ...

    def prompt_fragments(self, ctx: ToolBuildContext) -> list[str]:
        """本 provider 的 system prompt 片段（按返回顺序追加）。

        只有需要给模型定纪律的 provider 才有内容（目前是 web 与 kb）。
        返回空列表表示无片段。
        """
        ...
