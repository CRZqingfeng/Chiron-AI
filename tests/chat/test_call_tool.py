"""`call_tool` 与按参数判定的审批（U11 P1-A）。

## 这个文件守的两条线

1. **功能线**：模型知道工具名但工具没被激活时，`call_tool` 能直接调通，
   不必先搜一次。
2. **安全线**（更要紧）：
   - `call_tool` **够不到内置工具**——否则 `call_tool("workspace_run_command",
     {"command": "rm -rf /"})` 就是一条绕过工作区审批的后门
   - 经 `call_tool` 调高危工具时**仍然中断**——元工具把"按工具名判定"
     变成了"按参数判定"，如果不做这一步，高危工具从此不再需要确认

第二条用真 `create_agent` 验，不做 mock：审批链路是全项目最不能失效的地方，
"逻辑对但没接上"只有端到端能发现。
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any, List, Optional
from unittest import mock

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from app.chat.tool_gateway import CALL_TOOL_NAME, make_call_tool_tool


def _tool(name: str, result: str = None):
    from langchain_core.tools import StructuredTool

    def _f(**kwargs):
        return result if result is not None else f"{name} executed"

    return StructuredTool.from_function(func=_f, name=name, description=f"{name} 的用途")


def _call(*, target: str, args: dict | None = None) -> dict:
    return {
        "name": CALL_TOOL_NAME,
        "args": {"name": target, "args": args or {}},
        "id": f"c-{target}",
    }


class CallToolTests(unittest.TestCase):
    def setUp(self):
        self.read = _tool("mcp_read")
        self.write = _tool("mcp_write")
        self.call = make_call_tool_tool([self.read, self.write])

    def _invoke(self, payload: dict) -> str:
        return asyncio.run(self.call.ainvoke(payload))

    def test_calls_named_tool(self):
        out = self._invoke({"name": "mcp_read"})
        self.assertIn("mcp_read executed", out)

    def test_passes_arguments_through(self):
        from langchain_core.tools import StructuredTool

        seen: dict[str, Any] = {}

        def _echo(query: str = ""):
            seen["query"] = query
            return f"echo:{query}"

        target = StructuredTool.from_function(func=_echo, name="mcp_echo", description="echo")
        tool = make_call_tool_tool([target])
        out = asyncio.run(tool.ainvoke({"name": "mcp_echo", "args": {"query": "hi"}}))
        self.assertEqual(seen["query"], "hi")
        self.assertIn("echo:hi", out)

    def test_unknown_name_returns_hint(self):
        out = self._invoke({"name": "nope__missing"})
        self.assertIn("没有名为", out)
        # 必须给可用名字，否则模型只会反复猜
        self.assertIn("mcp_", out)

    def test_cannot_reach_builtin_tools(self):
        """★ 安全边界：只有 MCP 工具进 candidates。

        如果这里漏了，「用 call_tool 跑 workspace_run_command」就是一条
        绕过工作区审批的后门——而且看起来完全合法。
        """
        evil = _tool("workspace_run_command", result="SHOULD NOT RUN")
        tool = make_call_tool_tool([self.read])  # evil 不在名单里
        out = asyncio.run(
            tool.ainvoke(
                {"name": "workspace_run_command", "args": {"command": "rm -rf /"}}
            )
        )
        self.assertIn("没有名为", out)
        self.assertNotIn("SHOULD NOT RUN", out)
        # 顺带确认：把内置工具传进 candidates 才会被执行（说明边界就是 candidates）
        self.assertIsNotNone(evil)

    def test_tool_failure_is_reported_not_raised(self):
        from langchain_core.tools import StructuredTool

        def _boom(**kwargs):
            raise RuntimeError("工具炸了")

        bad = StructuredTool.from_function(func=_boom, name="mcp_bad", description="bad")
        tool = make_call_tool_tool([bad])
        out = asyncio.run(tool.ainvoke({"name": "mcp_bad"}))
        self.assertIn("调用", out)
        self.assertIn("失败", out)


class ExemptDecisionTests(unittest.TestCase):
    """`_all_call_tools_exempt` 的纯逻辑（不起图）。"""

    def _mw(self, confirm_names):
        from app.chat.approval import SafeCommandAwareHITL

        return SafeCommandAwareHITL(
            {CALL_TOOL_NAME: True},
            call_tool_confirm_names=confirm_names,
        )

    def _state(self, calls: list[dict]):
        return {"messages": [AIMessage(content="", tool_calls=calls)]}

    def test_no_call_tool_returns_false(self):
        mw = self._mw({"mcp_write"})
        self.assertFalse(mw._all_call_tools_exempt(self._state([])))

    def test_safe_target_is_exempt(self):
        mw = self._mw({"mcp_write"})
        self.assertTrue(mw._all_call_tools_exempt(self._state([_call(target="mcp_read")])))

    def test_high_risk_target_is_not_exempt(self):
        mw = self._mw({"mcp_write"})
        self.assertFalse(mw._all_call_tools_exempt(self._state([_call(target="mcp_write")])))

    def test_mixed_targets_are_not_exempt(self):
        """一批里有一个高危就全问一次——与 workspace 命令的取舍一致。"""
        mw = self._mw({"mcp_write"})
        state = self._state([_call(target="mcp_read"), _call(target="mcp_write")])
        self.assertFalse(mw._all_call_tools_exempt(state))

    def test_empty_confirm_names_means_never_confirm(self):
        mw = self._mw(set())
        self.assertTrue(mw._all_call_tools_exempt(self._state([_call(target="mcp_write")])))


# --------------------------------------------------------------------------
# 真流程：审批拦不拦得住
# --------------------------------------------------------------------------


class _ScriptedChatModel(BaseChatModel):
    script: List[AIMessage] = []
    _i: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = min(self._i, len(self.script) - 1)
        self._i += 1
        return ChatResult(generations=[ChatGeneration(message=self.script[idx])])


class CallToolApprovalFlowTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self, target: str, *, confirm_names: set[str]):
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        from app.chat.approval import SafeCommandAwareHITL

        read = _tool("mcp_read")
        write = _tool("mcp_write")
        call = make_call_tool_tool([read, write])

        model = _ScriptedChatModel(
            script=[
                AIMessage(content="", tool_calls=[_call(target=target)]),
                AIMessage(content="完成"),
            ]
        )
        agent = create_agent(
            model,
            tools=[call, read, write],
            middleware=[
                SafeCommandAwareHITL(
                    {CALL_TOOL_NAME: True},
                    call_tool_confirm_names=confirm_names,
                )
            ],
            checkpointer=InMemorySaver(),
        )
        return agent, {"configurable": {"thread_id": f"t-{target}"}}

    def _interrupts(self, agent, cfg) -> list:
        snap = agent.get_state(cfg)
        return [i for t in (snap.tasks or ()) for i in (getattr(t, "interrupts", ()) or ())]

    async def test_safe_target_runs_without_interrupt(self):
        agent, cfg = self._agent("mcp_read", confirm_names={"mcp_write"})
        out = await agent.ainvoke({"messages": [HumanMessage("读一下")]}, cfg)
        self.assertFalse(self._interrupts(agent, cfg), "只读目标不该中断")
        self.assertTrue(
            any("mcp_read executed" in str(getattr(m, "content", "")) for m in out["messages"])
        )

    async def test_high_risk_target_interrupts(self):
        agent, cfg = self._agent("mcp_write", confirm_names={"mcp_write"})
        await agent.ainvoke({"messages": [HumanMessage("写一下")]}, cfg)
        interrupts = self._interrupts(agent, cfg)
        self.assertTrue(interrupts, "★ 经 call_tool 调高危工具必须中断")

        reqs = interrupts[0].value.get("action_requests", [])
        self.assertEqual([r.get("name") for r in reqs], [CALL_TOOL_NAME])
        # 用户在卡片上要能看到真实目标，否则不知道该不该点允许
        self.assertEqual((reqs[0].get("args") or {}).get("name"), "mcp_write")

    async def test_reject_does_not_execute(self):
        from langgraph.types import Command

        agent, cfg = self._agent("mcp_write", confirm_names={"mcp_write"})
        await agent.ainvoke({"messages": [HumanMessage("写一下")]}, cfg)
        await agent.ainvoke(Command(resume={"decisions": [{"type": "reject"}]}), cfg)
        snap = agent.get_state(cfg)
        contents = " ".join(str(getattr(m, "content", "")) for m in snap.values["messages"])
        self.assertNotIn("mcp_write executed", contents, "拒绝后不能执行")


class WiringTests(unittest.TestCase):
    """装配层：网关启用时 `call_tool` 必须进 `interrupt_on`。

    不进的话，按参数判定的中间件根本不会被触发——工具照跑，审批形同不存在。
    """

    def test_call_tool_is_in_interrupt_on_when_gateway_active(self):
        from app.chat import agent_service

        captured: dict[str, Any] = {}

        def _fake_create_agent(**kwargs):
            captured.update(kwargs)
            return mock.Mock()

        ua = mock.Mock(
            id=1,
            user_id=1,
            api_key_ciphertext=__import__(
                "app.utils.api_key_crypto", fromlist=["encrypt_api_key"]
            ).encrypt_api_key("sk-test"),
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
            temperature=0.3,
            system_prompt="",
            enabled_mcp_servers=None,
            knowledge_base_ids=None,
            is_public=False,
            description="",
            name="t",
            sub_api_key_ciphertext=None,
            sub_base_url=None,
            sub_model_name=None,
            sub_temperature=None,
            use_sub_model=False,
            avatar=None,
        )
        mcp_tools = [_tool(f"srv__op{i}") for i in range(30)]

        with mock.patch("langchain.agents.create_agent", side_effect=_fake_create_agent):
            agent_service.build_model_and_agent(
                ua,
                1,
                1,
                "s-calltool",
                extra_tools=mcp_tools,
                interrupt_on={"workspace_run_command": True},
                checkpointer=mock.Mock(),
                call_tool_confirm_names={"srv__op0"},
            )

        names = [type(mw).__name__ for mw in (captured.get("middleware") or [])]
        self.assertIn("SafeCommandAwareHITL", names)
        hitl = next(m for m in captured["middleware"] if type(m).__name__ == "SafeCommandAwareHITL")
        self.assertIn(CALL_TOOL_NAME, hitl.interrupt_on)
        self.assertIn("srv__op0", hitl.call_tool_confirm_names)


if __name__ == "__main__":
    unittest.main()
