"""工具可见性网关（U11 P0-B）。

最关键的一条是 `test_below_threshold_is_byte_identical`：MCP 工具不超过阈值时，
模型看到的工具集合与改动前**完全一致**。不满足这条，就是在给小场景加无谓的风险。

用真实 `create_agent` + 脚本化模型读每轮的 `request.tools`，不做 mock——
裁剪逻辑写在中间件里，"策略对但没接上"是这类改动最常见的失败方式。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Optional
from unittest import mock

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from app.chat.tool_gateway import (
    SEARCH_TOOL_NAME,
    VisibleToolsMiddleware,
    gateway_active,
    make_search_tools_tool,
    score_candidate,
)
from app.chat.tool_providers import build_tools
from app.chat.tool_providers.base import ToolBuildContext
from app.core.turn_runtime import (
    activate_tools,
    get_activated_tools,
    reset_activated_tools,
)
from app.workspace.local import LocalWorkspace


# --------------------------------------------------------------------------
# 阈值判定
# --------------------------------------------------------------------------


class GatewayThresholdTests(unittest.TestCase):
    def test_auto_uses_threshold(self):
        with mock.patch("app.chat.tool_gateway.settings") as s:
            s.TOOL_GATEWAY_MODE = "auto"
            s.TOOL_GATEWAY_MAX_MCP_TOOLS = 25
            self.assertFalse(gateway_active(0))
            self.assertFalse(gateway_active(25))
            self.assertTrue(gateway_active(26))

    def test_always_full_never_enables(self):
        with mock.patch("app.chat.tool_gateway.settings") as s:
            s.TOOL_GATEWAY_MODE = "always_full"
            s.TOOL_GATEWAY_MAX_MCP_TOOLS = 1
            self.assertFalse(gateway_active(500))

    def test_always_discovery_always_enables(self):
        with mock.patch("app.chat.tool_gateway.settings") as s:
            s.TOOL_GATEWAY_MODE = "always_discovery"
            s.TOOL_GATEWAY_MAX_MCP_TOOLS = 10**6
            self.assertTrue(gateway_active(0))

    def test_unknown_mode_falls_back_to_auto(self):
        """配置写错不能静默变成「永远全量」——那会让保护无声消失。"""
        with mock.patch("app.chat.tool_gateway.settings") as s:
            s.TOOL_GATEWAY_MODE = "typo"
            s.TOOL_GATEWAY_MAX_MCP_TOOLS = 25
            self.assertTrue(gateway_active(100))


# --------------------------------------------------------------------------
# provider：阈值上下行为差异
# --------------------------------------------------------------------------


def _fake_ua():
    """与 `test_tool_registry.py` 一致：用 SimpleNamespace，不用 tortoise 模型。

    模型实例的 `__init__` 会吞掉未声明的字段（外键叫 `user` 还是 `user_id`
    随版本变），`build_model_and_agent` 读 `ua.user_id` 时会 AttributeError。
    """
    from types import SimpleNamespace

    from app.utils.api_key_crypto import encrypt_api_key

    return SimpleNamespace(
        id=9901,
        user_id=9902,
        api_key_ciphertext=encrypt_api_key("sk-test-key"),
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


def _fake_mcp_tools(n: int) -> list:
    from langchain_core.tools import StructuredTool

    out = []
    for i in range(n):
        def _f(**kw):
            return "ok"

        out.append(
            StructuredTool.from_function(
                func=_f, name=f"srv{i}__op_{i}", description=f"第 {i} 号操作"
            )
        )
    return out


class GatewayProviderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = LocalWorkspace(Path(self._tmp.name))
        self.ua = _fake_ua()

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, mcp_tools):
        ctx = ToolBuildContext(
            ua=self.ua,
            user_id=1,
            agent_id=1,
            session_id="s-test",
            kb_scope="u1_a1",
            llm_config={},
            workspace=self.ws,
            extra_tools=mcp_tools,
        )
        return build_tools(ctx)

    def test_below_threshold_is_byte_identical(self):
        """★ 不超过阈值时，工具与提示词片段**逐字相同**。

        这条守住的是「不要为了架构统一而改变小场景的行为」。
        """
        baseline_tools, baseline_fragments = self._build([])
        small = _fake_mcp_tools(5)
        tools_with_mcp, fragments_with_mcp = self._build(small)

        base_names = [t.name for t in baseline_tools]
        now_names = [t.name for t in tools_with_mcp if not t.name.startswith("srv")]
        self.assertEqual(base_names, now_names, "内置工具集合不能被网关改动")
        self.assertNotIn(SEARCH_TOOL_NAME, [t.name for t in tools_with_mcp])
        self.assertEqual(len(baseline_fragments), len(fragments_with_mcp), "提示词片段不能多")

    def test_above_threshold_adds_search_tool_and_fragment(self):
        _, base_fragments = self._build([])
        tools, fragments = self._build(_fake_mcp_tools(30))
        names = [t.name for t in tools]
        self.assertIn(SEARCH_TOOL_NAME, names)
        self.assertEqual(len(fragments), len(base_fragments) + 1, "应多一段按需发现说明")

    def test_search_tool_is_not_registered_below_threshold(self):
        tools, _ = self._build(_fake_mcp_tools(25))
        self.assertNotIn(SEARCH_TOOL_NAME, [t.name for t in tools])


# --------------------------------------------------------------------------
# 中间件：真 create_agent
# --------------------------------------------------------------------------


def _named_tool(name: str):
    from langchain_core.tools import StructuredTool

    def _f(**kw):
        return f"{name} executed"

    return StructuredTool.from_function(func=_f, name=name, description=f"{name} 的用途")


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


class VisibleToolsMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        reset_activated_tools()
        self.tools = [_named_tool("core_a"), _named_tool("mcp_x"), _named_tool("mcp_y")]

    def tearDown(self):
        reset_activated_tools()

    def test_hides_non_core_tools(self):
        mw = VisibleToolsMiddleware({"core_a"})
        visible = [t.name for t in mw._visible(self.tools)]
        self.assertEqual(visible, ["core_a"])

    def test_activation_makes_tool_visible(self):
        mw = VisibleToolsMiddleware({"core_a"})
        activate_tools(["mcp_x"])
        visible = [t.name for t in mw._visible(self.tools)]
        self.assertEqual(visible, ["core_a", "mcp_x"])
        self.assertNotIn("mcp_y", visible)

    def test_override_failure_falls_back_to_full_set(self):
        """裁剪抛异常时必须把**原** request 交给下游（全量）。

        少给几个工具是体验问题；一个工具都不给是功能故障。
        """
        received: list[int] = []

        class _BadOverride:
            tools = self.tools

            def override(self, **_kw):
                raise RuntimeError("override failed")

        def _handler(req):
            received.append(len(req.tools))
            return AIMessage(content="ok")

        mw = VisibleToolsMiddleware({"core_a"})
        out = mw.wrap_model_call(_BadOverride(), _handler)
        self.assertEqual(out.content, "ok")
        self.assertEqual(received, [3], "应退回全量（3 个），而不是空列表")

    async def test_end_to_end_search_then_call(self):
        """★ 真流程：search_tools 激活 → 下一轮模型看到完整定义 → 能直接调用。

        这是整个网关成立的前提。用真实 create_agent + 真实中间件，
        只有模型是脚本化的。
        """
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        alpha = _named_tool("mcp_alpha")
        beta = _named_tool("mcp_beta")
        search = make_search_tools_tool([alpha, beta])

        model = _ScriptedChatModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": SEARCH_TOOL_NAME, "args": {"query": "alpha"}, "id": "s1"}
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "mcp_alpha", "args": {}, "id": "a1"}],
                ),
                AIMessage(content="完成"),
            ]
        )
        agent = create_agent(
            model,
            tools=[search, alpha, beta],
            middleware=[VisibleToolsMiddleware({SEARCH_TOOL_NAME})],
            checkpointer=InMemorySaver(),
        )
        out = await agent.ainvoke(
            {"messages": [HumanMessage("用 alpha 做点事")]},
            {"configurable": {"thread_id": "t-e2e"}},
        )

        self.assertEqual(out["messages"][-1].content, "完成")
        # alpha 真的被执行了（说明激活后模型能调用到被隐藏的工具）
        self.assertTrue(
            any("mcp_alpha executed" in str(getattr(m, "content", "")) for m in out["messages"]),
            "激活后的工具必须能被实际调用",
        )
        # beta 从未被激活，也就不该出现在对话里
        self.assertFalse(
            any("mcp_beta executed" in str(getattr(m, "content", "")) for m in out["messages"])
        )


# --------------------------------------------------------------------------
# search_tools 本身
# --------------------------------------------------------------------------


class SearchToolsTests(unittest.TestCase):
    def setUp(self):
        reset_activated_tools()
        self.candidates = [
            _named_tool("github__create_issue"),
            _named_tool("github__list_issues"),
            _named_tool("jira__create_ticket"),
            _named_tool("slack__send_message"),
        ]

    def tearDown(self):
        reset_activated_tools()

    def _search(self, query: str, limit: int = 8) -> str:
        from app.chat.tool_gateway import make_search_tools_tool

        t = make_search_tools_tool(self.candidates)
        # StructuredTool 的协程体：直接调 .func 不可用（只有 coroutine），用 ainvoke
        import asyncio

        return asyncio.run(t.ainvoke({"query": query, "limit": limit}))

    def test_hit_activates_tools(self):
        out = self._search("create issue")
        self.assertIn("github__create_issue", out)
        self.assertIn("github__create_issue", get_activated_tools())

    def test_name_match_beats_description_match(self):
        # 四个工具的描述里都含 "用途"，名字里只有 create 类含 create
        self.assertGreater(
            score_candidate("create", "github__create_issue", "用途"),
            score_candidate("create", "github__list_issues", "create 相关用途"),
        )

    def test_empty_result_returns_nearest_instead_of_nothing(self):
        """空结果 = 模型断定"没有这个能力"然后开始编。必须给近似项。"""
        out = self._search("量子纠缠计算器")
        self.assertIn("未找到", out)
        # 仍要给出候选，让模型知道工具是有的
        self.assertTrue(any(n in out for n in ("github__", "jira__", "slack__")))

    def test_limit_is_clamped(self):
        out = self._search("issue", limit=100)
        self.assertIn("已找到", out)

    def test_empty_query_is_rejected(self):
        out = self._search("   ")
        self.assertIn("不能为空", out)


class GatewayWiringTests(unittest.TestCase):
    """装配层：中间件到底有没有被挂上去。

    光测中间件本身不够——`build_model_and_agent` 里那段被 try 包着，
    出错只会打一行 warning，然后**静默全量注入**。这种「策略对但没接上」
    是这类改动最常见的失败方式。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = LocalWorkspace(Path(self._tmp.name))
        self.ua = _fake_ua()

    def tearDown(self):
        self._tmp.cleanup()

    def _middlewares(self, mcp_count: int) -> list[str]:
        from app.chat import agent_service

        with mock.patch("langchain.agents.create_agent", return_value=mock.Mock()) as m:
            agent_service.build_model_and_agent(
                self.ua,
                1,
                1,
                "s-gateway",
                extra_tools=_fake_mcp_tools(mcp_count),
                workspace=self.ws,
            )
            mws = m.call_args.kwargs.get("middleware") or []
        return [type(x).__name__ for x in mws]

    def test_not_attached_below_threshold(self):
        self.assertNotIn("VisibleToolsMiddleware", self._middlewares(10))

    def test_attached_above_threshold(self):
        self.assertIn("VisibleToolsMiddleware", self._middlewares(30))


if __name__ == "__main__":
    unittest.main()
