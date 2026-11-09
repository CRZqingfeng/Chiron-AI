"""工具 provider 的装配契约（U6b）。

守四件事：

1. **provider 组合的工具数与改造前一致** —— 加/少工具会让模型行为变化
2. **system prompt 与旧路径逐字节一致** —— 提示词是模型行为的直接输入，
   改一个字都可能改变输出，所以这里比"看起来差不多"更严格
3. **片段顺序为「联网 → KB 图片 → KB 作答」** —— 决定模型先读到哪条纪律
4. **MCP 工具走同一装配流程**，不是单独 extend

基线见 `docs/dev/U6-baseline.md`（由 `.u6_baseline.py` 生成，人工不可编辑）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.chat.agent_prompt import _compose_system_prompt, _format_turn_context_block
from app.chat.tool_providers import TOOL_PROVIDERS, ToolBuildContext, build_tools
from app.utils.api_key_crypto import encrypt_api_key
from app.workspace.local import LocalWorkspace


def _fake_ua() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        user_id=1,
        api_key_ciphertext=encrypt_api_key("sk-test"),
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o-mini",
        temperature=0.3,
        system_prompt="你是一个严谨的助手。",
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


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = LocalWorkspace(Path(self._tmp.name))
        self.ua = _fake_ua()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def ctx(self, **overrides) -> ToolBuildContext:
        base = dict(
            ua=self.ua,
            user_id=1,
            agent_id=1,
            session_id="s-test",
            kb_scope="u1_a1",
            llm_config={},
            workspace=self.ws,
        )
        base.update(overrides)
        return ToolBuildContext(**base)

    def names(self, **overrides) -> list[str]:
        tools, _ = build_tools(self.ctx(**overrides))
        return sorted(t.name for t in tools)


class ProviderRegistryTests(unittest.TestCase):
    def test_providers_sorted_by_order(self):
        orders = [p.order for p in TOOL_PROVIDERS]
        self.assertEqual(orders, sorted(orders), "TOOL_PROVIDERS 必须按 order 有序")

    def test_web_before_kb(self):
        """prompt 片段顺序依赖这个：联网纪律在前，知识库纪律在后。"""
        names = [p.name for p in TOOL_PROVIDERS]
        self.assertLess(names.index("web"), names.index("kb"))

    def test_all_providers_implement_protocol(self):
        for p in TOOL_PROVIDERS:
            for attr in ("enabled", "build", "prompt_fragments"):
                self.assertTrue(callable(getattr(p, attr, None)), f"{p.name} 缺 {attr}")


class ToolCountTests(_Base):
    """T4~T7：组合后的工具集合数量。

    数字会随工具增减而变，变更时必须同步改这里——这是刻意的：
    数量断言的意义是「让新增工具必须被显式确认」，而不是「永远等于某个魔数」。
    U7 加了 search_long_term_memory，各档位 +1（12→13 / 9→10 / 13→14）。
    U10 P0 加了 workspace_edit_file / grep / glob，带工作区的档位 +3（13→16 / 14→17）；
    无工作区档位不变（10）。
    """

    def test_default_with_workspace(self):
        names = self.names()
        self.assertEqual(len(names), 16)
        self.assertIn("search_long_term_memory", names)
        self.assertIn("workspace_write_file", names)
        self.assertIn("workspace_edit_file", names)
        self.assertIn("workspace_grep", names)
        self.assertIn("workspace_glob", names)
        # 没有 interrupt_on 时不注册执行命令工具（安全约束）
        self.assertNotIn("workspace_run_command", names)

    def test_without_workspace(self):
        names = self.names(workspace=None)
        self.assertEqual(len(names), 10)
        self.assertFalse([n for n in names if n.startswith("workspace_")])

    def test_with_approval_registers_exec(self):
        names = self.names(interrupt_on={"workspace_run_command": True})
        self.assertEqual(len(names), 17)
        self.assertIn("workspace_run_command", names)

    def test_web_disabled(self):
        with mock.patch("app.settings.settings.WEB_SEARCH_ENABLED", False):
            names = self.names()
        self.assertNotIn("web_search", names)
        self.assertNotIn("fetch_url", names)
        # 知识库与记忆不受影响
        self.assertIn("search_knowledge_base", names)
        self.assertIn("search_session_memory", names)

    def test_memory_disabled(self):
        with mock.patch("app.settings.settings.CHAT_USE_SESSION_MEMORY", False):
            names = self.names()
        self.assertNotIn("search_session_memory", names)
        # 关掉会话记忆不该连带关掉长期记忆：两者触发条件与开关都独立
        self.assertIn("search_long_term_memory", names)

    def test_long_term_memory_disabled_keeps_session_memory(self):
        with mock.patch("app.settings.settings.CHAT_MEMORY_CURATION_ENABLED", False):
            names = self.names()
        self.assertNotIn("search_long_term_memory", names)
        self.assertIn("search_session_memory", names)

    def test_extra_tools_flow_through_mcp_provider(self):
        """MCP 工具必须走同一装配流程，不是单独 extend。"""
        fake = mock.Mock()
        fake.name = "mcp_demo_tool"
        names = self.names(extra_tools=[fake])
        self.assertIn("mcp_demo_tool", names)


class PromptFidelityTests(_Base):
    """T8：新路径与旧路径的 system prompt 必须逐字节一致。"""

    CONFIGS = [
        ("kb_only", True, False),
        ("kb_web", True, True),
        ("web_only", False, True),
        ("neither", False, False),
    ]

    def test_new_matches_legacy(self):
        for label, kb, web in self.CONFIGS:
            with self.subTest(config=label):
                ctx = self.ctx(use_knowledge_retrieval=kb, use_web_search=web)
                _, fragments = build_tools(ctx)
                new = _compose_system_prompt(self.ua, fragments=fragments)
                old = _compose_system_prompt(
                    self.ua, use_knowledge_retrieval=kb, use_web_search=web
                )
                self.assertEqual(new, old)

    def test_fragment_order(self):
        """联网纪律必须排在知识库纪律之前（对拍已保证内容，这里守顺序意图）。"""
        _, fragments = build_tools(self.ctx(use_knowledge_retrieval=True, use_web_search=True))
        self.assertEqual(len(fragments), 3)
        self.assertIn("联网搜索纪律", fragments[0])
        self.assertIn("知识库检索结果中", fragments[1])
        self.assertIn("知识库作答纪律", fragments[2])

    def test_no_fragments_when_both_off(self):
        _, fragments = build_tools(self.ctx(use_knowledge_retrieval=False, use_web_search=False))
        self.assertEqual(fragments, [])
        self.assertEqual(
            _compose_system_prompt(self.ua, fragments=fragments),
            (self.ua.system_prompt or "").strip(),
        )

    def test_turn_context_block_unchanged(self):
        """T9：turn context 本轮结构未改，但守一下别在后续改动里漂移。"""
        block = _format_turn_context_block(
            use_knowledge_retrieval=True,
            use_web_search=False,
            document_filter=None,
            session_attachment_hint="",
            memory_inject=None,
            image_caption=None,
        )
        self.assertIn("本轮允许调用 search_knowledge_base", block)
        self.assertIn("不要调用 web_search", block)


class LegacyPathTests(unittest.TestCase):
    """旧装配路径必须仍可用（回退开关），且产出与新路径一致。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = LocalWorkspace(Path(self._tmp.name))
        self.ua = _fake_ua()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _names_via_build(self, **kwargs) -> set[str]:
        from app.chat import agent_service

        with mock.patch("langchain.agents.create_agent", return_value=mock.Mock()) as m:
            agent_service.build_model_and_agent(self.ua, 1, 1, "s1", **kwargs)
            return {t.name for t in (m.call_args.kwargs.get("tools") or [])}

    def test_switch_both_ways_same_tools(self):
        with mock.patch("app.settings.settings.CHAT_TOOL_REGISTRY_ENABLED", True):
            new = self._names_via_build(workspace=self.ws)
        with mock.patch("app.settings.settings.CHAT_TOOL_REGISTRY_ENABLED", False):
            old = self._names_via_build(workspace=self.ws)
        self.assertEqual(new, old)

    def test_switch_both_ways_same_prompt(self):
        from app.chat import agent_service

        def prompt_for(enabled: bool) -> str:
            with mock.patch("app.settings.settings.CHAT_TOOL_REGISTRY_ENABLED", enabled):
                with mock.patch(
                    "langchain.agents.create_agent", return_value=mock.Mock()
                ) as m:
                    agent_service.build_model_and_agent(self.ua, 1, 1, "s1", workspace=self.ws)
                    return m.call_args.kwargs.get("system_prompt")

        self.assertEqual(prompt_for(True), prompt_for(False))


if __name__ == "__main__":
    unittest.main()
