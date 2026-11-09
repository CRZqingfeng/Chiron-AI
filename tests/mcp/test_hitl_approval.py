"""MCP 高危工具走 HITL 审批（U5c）。

三个层次，各自验证不同的东西：

- `ConfirmationPolicyTests`：`should_confirm_mcp_tool` 的三档策略（纯函数）
- `ConfirmationGateTests`：高危工具如何接进审批链路（有/无 checkpointer 两条分支）
- `McpHitlFlowTests`：**真** `create_agent` + `InMemorySaver`，验证「断点续跑」

第三条是 U5c 的全部意义所在。它断言的不是"会不会中断"（那谁都看得出来），
而是"中断前已经执行过的工具**不会**在恢复后重跑一遍"——
U5c 之前 MCP 走「批准票 + 整轮重跑」，一次 `list_repos` 会被执行两次。
"""

from __future__ import annotations

import unittest
from typing import Any, List, Optional
from unittest import mock

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import PrivateAttr


# --------------------------------------------------------------------------
# 判定：三档 confirm_policy
# --------------------------------------------------------------------------


def _fake_tool(name: str, description: str = "", annotations: dict | None = None):
    """造一个带 annotations 的工具替身（只用到 name / description / annotations）。"""

    class _T:
        pass

    t = _T()
    t.name = name
    t.description = description
    if annotations is not None:
        t.annotations = annotations
    return t


class ConfirmationPolicyTests(unittest.TestCase):
    def setUp(self):
        from app.mcp_client import tool_policy

        self.policy = tool_policy
        # 默认开关是打开的；个别用例里手动关掉
        p = mock.patch.object(tool_policy, "confirmation_required", lambda: True)
        p.start()
        self.addCleanup(p.stop)

    # ---- 策略档位 ----
    def test_never_policy_never_confirms(self):
        t = _fake_tool("delete_everything", "删库")
        self.assertFalse(self.policy.should_confirm_mcp_tool("srv", t, "never"))

    def test_always_policy_always_confirms(self):
        t = _fake_tool("list_issues", "只读列表")
        self.assertTrue(self.policy.should_confirm_mcp_tool("srv", t, "always"))

    def test_auto_uses_annotations_first(self):
        read_only = _fake_tool("write_thing", "写入", {"readOnlyHint": True})
        destructive = _fake_tool("get_thing", "读取", {"destructiveHint": True})
        # annotations 优先级高于名字里的动词：用了就照它走
        self.assertFalse(self.policy.should_confirm_mcp_tool("srv", read_only, "auto"))
        self.assertTrue(self.policy.should_confirm_mcp_tool("srv", destructive, "auto"))

    def test_auto_falls_back_to_keywords(self):
        # 没有 annotations 时按动词猜，这是启发式，默认保守
        self.assertTrue(self.policy.should_confirm_mcp_tool("srv", _fake_tool("create_issue"), "auto"))
        self.assertTrue(self.policy.should_confirm_mcp_tool("srv", _fake_tool("delete_repo"), "auto"))
        self.assertFalse(self.policy.should_confirm_mcp_tool("srv", _fake_tool("get_me"), "auto"))

    def test_global_switch_off_disables_everything(self):
        with mock.patch.object(self.policy, "confirmation_required", lambda: False):
            t = _fake_tool("delete_repo")
            self.assertFalse(self.policy.should_confirm_mcp_tool("srv", t, "always"))

    def test_annotations_read_from_metadata(self):
        """adapters 0.3.x 把 annotations **平铺**进 metadata（见 tools.py:510-513）。

        这条曾经是坏的：旧实现只认 `metadata["annotations"]` 嵌套形态，
        于是 `readOnlyHint` 被静默丢弃，只读工具只能靠动词猜。
        """
        t = _fake_tool("anything", "描述")
        t.metadata = {"readOnlyHint": True, "_meta": {"x": 1}}
        self.assertFalse(self.policy.should_confirm_mcp_tool("srv", t, "auto"))

    def test_annotations_nested_forms_still_work(self):
        nested = _fake_tool("anything", "描述")
        nested.metadata = {"annotations": {"destructiveHint": True}}
        self.assertTrue(self.policy.should_confirm_mcp_tool("srv", nested, "auto"))

        deep = _fake_tool("anything", "描述")
        deep.metadata = {"mcp": {"annotations": {"readOnlyHint": True}}}
        self.assertFalse(self.policy.should_confirm_mcp_tool("srv", deep, "auto"))

    def test_unrelated_metadata_is_not_mistaken_for_annotations(self):
        t = _fake_tool("create_issue", "创建 issue")
        t.metadata = {"version": "1.0", "owner": "team"}
        # 认不出 annotations → 退回关键词，create 命中 → 要确认
        self.assertTrue(self.policy.should_confirm_mcp_tool("srv", t, "auto"))


# --------------------------------------------------------------------------
# 接入：有/无 checkpointer
# --------------------------------------------------------------------------


class ConfirmationGateTests(unittest.TestCase):
    def _tool(self, name: str):
        class _T:
            pass

        t = _T()
        t.name = name
        return t

    def test_with_checkpointer_merges_into_interrupt_on(self):
        from app.chat.approval import apply_mcp_confirmation_gate

        tools = [self._tool("mcp_read"), self._tool("mcp_write")]
        base = {"workspace_run_command": True}
        merged, kept = apply_mcp_confirmation_gate(
            base, tools, {"mcp_write"}, checkpointer_available=True
        )
        self.assertEqual(merged, {"workspace_run_command": True, "mcp_write": True})
        self.assertEqual([t.name for t in kept], ["mcp_read", "mcp_write"])
        # 不能改调用方传进来的 dict：同一个 interrupt_on 可能在别处复用
        self.assertEqual(base, {"workspace_run_command": True})

    def test_without_checkpointer_drops_high_risk_tools(self):
        """没有中断能力时必须不注册——保留即等于零确认放行（安全倒退）。"""
        from app.chat.approval import apply_mcp_confirmation_gate

        tools = [self._tool("mcp_read"), self._tool("mcp_write")]
        merged, kept = apply_mcp_confirmation_gate(
            {}, tools, {"mcp_write"}, checkpointer_available=False
        )
        self.assertEqual(merged, {})
        self.assertEqual([t.name for t in kept], ["mcp_read"])

    def test_empty_confirm_names_is_a_noop(self):
        from app.chat.approval import apply_mcp_confirmation_gate

        tools = [self._tool("mcp_read")]
        base = {"x": True}
        merged, kept = apply_mcp_confirmation_gate(
            base, tools, set(), checkpointer_available=False
        )
        self.assertIs(merged, base)
        self.assertEqual(kept, tools)


# --------------------------------------------------------------------------
# 真流程：断点续跑
# --------------------------------------------------------------------------


class _ScriptedChatModel(BaseChatModel):
    """按脚本依次返回 AIMessage。`bind_tools` 必须返回自身，否则 create_agent 会报错。"""

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


class McpHitlFlowTests(unittest.TestCase):
    """真实中间件 + 真实图，唯一替身是模型和工具实现（它们要计数）。"""

    def _build(self, calls: dict):
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        from app.chat.approval import SafeCommandAwareHITL

        @tool
        def mcp_list_repos() -> str:
            """列出仓库（只读，不审批）。"""
            calls["list"] += 1
            return f"repos 第 {calls['list']} 次"

        @tool
        def mcp_create_issue(title: str) -> str:
            """创建 issue（高危，要审批）。"""
            calls["create"] += 1
            return f"created {title} 第 {calls['create']} 次"

        model = _ScriptedChatModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "mcp_list_repos", "args": {}, "id": "c1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "mcp_create_issue", "args": {"title": "bug"}, "id": "c2"}
                    ],
                ),
                AIMessage(content="做完了"),
            ]
        )
        agent = create_agent(
            model,
            tools=[mcp_list_repos, mcp_create_issue],
            middleware=[SafeCommandAwareHITL(interrupt_on={"mcp_create_issue": True})],
            checkpointer=InMemorySaver(),
        )
        return agent, {"configurable": {"thread_id": "t-mcp"}}

    def test_interrupt_then_approve_does_not_rerun_earlier_tools(self):
        from langgraph.types import Command

        calls = {"list": 0, "create": 0}
        agent, cfg = self._build(calls)

        agent.invoke({"messages": [HumanMessage("先列仓库再建 issue")]}, cfg)
        # 第一轮：list 已执行，create 被拦
        self.assertEqual(calls, {"list": 1, "create": 0})

        snap = agent.get_state(cfg)
        interrupts = [i for t in (snap.tasks or ()) for i in (getattr(t, "interrupts", ()) or ())]
        self.assertTrue(interrupts, "高危 MCP 工具必须中断")
        names = [a.get("name") for a in interrupts[0].value.get("action_requests", [])]
        self.assertEqual(names, ["mcp_create_issue"])

        out = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg)
        # ★ U5c 的核心收益：list 没被重跑
        self.assertEqual(calls, {"list": 1, "create": 1})
        self.assertEqual(out["messages"][-1].content, "做完了")

    def test_reject_does_not_execute_tool(self):
        from langgraph.types import Command

        calls = {"list": 0, "create": 0}
        agent, cfg = self._build(calls)

        agent.invoke({"messages": [HumanMessage("先列仓库再建 issue")]}, cfg)
        agent.invoke(Command(resume={"decisions": [{"type": "reject"}]}), cfg)

        self.assertEqual(calls["create"], 0, "拒绝后不能执行")
        self.assertEqual(calls["list"], 1)


if __name__ == "__main__":
    unittest.main()
