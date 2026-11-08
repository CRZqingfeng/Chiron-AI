"""审批中间件：只读命令免人工确认，其余照旧中断。

## 为什么需要一层中间件

LangChain 的 `HumanInTheLoopMiddleware` **只按工具名判定**要不要中断：

```python
for idx, tool_call in enumerate(last_ai_msg.tool_calls):
    if (config := self.interrupt_on.get(tool_call["name"])) is not None:
        ...  # 进中断
```

`interrupt_on` 的值只有「允许哪些决定」，没有「按参数再判断」的钩子。所以
`workspace_run_command` 一旦登记进来，连 `ls -la` 都要用户点一次。

## 做法：包一层，只改「要不要中断」这一件事

`after_model` 里先判断「这条 AI 消息里所有命令类调用是否**全部**可证明只读」：

- 全部只读 → 从 `interrupt_on` 里摘掉 `workspace_run_command`，再交给官方中间件
- 有任何一条不安全 → 原样交给官方中间件，**全部命令调用一起中断**

### 为什么是「包一层」而不是「继承后重写父类方法」

第一版是 `class SafeCommandAwareHITL(HumanInTheLoopMiddleware)`，覆写
`after_model` 后委托一个新实例。功能是对的，但**类定义绑定了导入时机**：
`tests/chat/test_todo_planning.py` 里有个用例会 `mock.patch` 官方中间件来模拟
「装配失败」，如果 `app.chat.approval` 恰好在那次 patch 期间首次被导入，本类就会
继承到一个 Mock ——单跑通过、全量跑报 `StopIteration`。这类问题只在特定执行顺序下
出现，最难排查。

所以改成继承 `AgentMiddleware`（不会被 patch 的稳定基类），并在**方法内部**延迟
导入官方中间件。导入时机不再影响类型的正确性。

### 为什么不做「逐条只放行安全的」

那需要重写官方 `after_model` 里 90 行的循环（构造 action_requests、对齐
interrupt_indices、按序重建 tool_calls、校验 decisions 数量）。这段一旦抄进本仓，
langchain 升级就会在无提示的情况下与上游语义分叉——U5b 的审批链路是全项目最不能
出错的地方，不值得为「一条消息里同时有安全和不安全命令」这种罕见组合去冒风险。
代价只是：这种情况下安全命令也会一起被问一次（保守方向）。

## 降级

策略函数抛异常时**一律走原路径**（全部要确认）。宁可多问一次，不可静默放行。

## 测试类命令的会话级授权（U10 P1-E）

判定从「全部只读」扩展成「全部可免确认」：可证明只读 **或**
（测试类命令 **且** 本会话已批准过一次）。

授权状态存在 Redis（`app.workspace.command_grants`），key 带「用户×智能体×会话」作用域。
作用域来自 `app.core.turn_runtime` 的 ContextVar —— 中间件只能拿到 state / runtime，
而 `thread_id` 是 **job_id 不是 session_id**，ContextVar 是唯一能拿到会话的通道。

**为什么不无条件放行 pytest**：它会加载 `conftest.py`，而那是模型可写的文件。
无条件免审批等于开了一条「写 conftest.py + pytest」绕过 `python script.py` 审批的后门。
所以必须有「用户批准过」这个前置。
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage

from app.log import logger
from app.workspace.command_policy import SAFE_COMMAND_TOOL, is_safe_readonly_command, is_test_command


def _test_commands_granted() -> bool:
    """本会话是否已授权跑测试类命令（延迟导入：chat 与 workspace 无模块级硬依赖）。"""
    try:
        from app.core.turn_runtime import get_session_scope
        from app.workspace.command_grants import test_commands_granted

        return test_commands_granted(get_session_scope())
    except Exception:  # noqa: BLE001 - 判定故障按未授权处理
        return False


class SafeCommandAwareHITL(AgentMiddleware):
    """只读命令免确认的审批中间件（`CHAT_APPROVAL_SAFE_COMMANDS_ENABLED` 控制启停）。"""

    def __init__(
        self,
        interrupt_on: dict[str, Any],
        *,
        description_prefix: str = "Tool execution requires approval",
    ) -> None:
        super().__init__()
        # 保留原始配置（含 bool 值），构造官方中间件时再让它自己规范化
        self.interrupt_on: dict[str, Any] = dict(interrupt_on or {})
        self.description_prefix = description_prefix

    # -- 判定 ---------------------------------------------------------------

    def _all_commands_exempt(self, state: dict[str, Any]) -> bool:
        """本轮所有 `workspace_run_command` 调用是否都可免确认。

        两种免确认：
        1. 可证明只读（`ls` / `cat` / `git status` …）
        2. 测试类命令，**且本会话已经批准过一次**（U10 P1-E）

        第 2 条为什么需要「批准过一次」：pytest 会加载 `conftest.py`，那是模型可写的文件。
        无条件放行等于给了一条绕过 `python script.py` 审批的通道。

        :return: True 表示可以整体跳过命令类审批；没有命令调用时返回 False
            （没有命令就没有需要跳过的，交给官方中间件处理其他工具）
        """
        messages = state.get("messages") or []
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai is None or not getattr(last_ai, "tool_calls", None):
            return False

        commands = [
            str(((tc.get("args") or {}).get("command")) or "")
            for tc in last_ai.tool_calls
            if tc.get("name") == SAFE_COMMAND_TOOL
        ]
        if not commands:
            return False

        granted: bool | None = None
        for command in commands:
            if is_safe_readonly_command(command)[0]:
                continue
            if granted is None:
                # 只在真有非只读命令时才查一次 Redis，别为纯只读轮次多一次往返
                granted = _test_commands_granted()
            if granted and is_test_command(command)[0]:
                continue
            return False
        return True

    # -- 钩子 ---------------------------------------------------------------

    def _build_delegate(self, state: dict[str, Any]) -> AgentMiddleware:
        """按「本轮命令是否都可免确认」构造一个官方中间件实例。

        每次都新建：便宜（只是个配置容器），且不会修改自身状态，天然并发安全。
        """
        # 延迟导入：见模块 docstring「为什么是包一层」
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        config = dict(self.interrupt_on)
        try:
            if self._all_commands_exempt(state):
                config.pop(SAFE_COMMAND_TOOL, None)
        except Exception as e:  # noqa: BLE001 - 判定故障必须退回「全部要确认」
            logger.warning("命令免确认判定失败，本轮恢复为全部需确认：{}", e)
        return HumanInTheLoopMiddleware(
            interrupt_on=config,
            description_prefix=self.description_prefix,
        )

    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        return self._build_delegate(state).after_model(state, runtime)

    async def aafter_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        # 官方中间件的 aafter_model 也只是转调同步实现，这里保持一致
        return self.after_model(state, runtime)
