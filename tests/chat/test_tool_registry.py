"""工具注册表的契约测试（U6a）。

守三个不变量：

1. **注册表覆盖全部实际注册的工具** —— 新增工具忘了登记，这条会红。
   改造前没人守这一条，于是 U4 加的 4 个 `workspace_*` 漏在 MCP 撞名白名单外，
   MCP 可以静默顶替它们（`create_agent` 对重名工具不报错，已实测）。
2. **重名必须抛错**，不能静默保留其一。
3. **白名单从注册表派生**，不是手工另抄一份。

做法沿用 `tests/workspace/test_wiring.py`：拦截 `create_agent` 抓它实际收到的
tools。比翻 `CompiledStateGraph` 内部结构可靠（后者是私有实现，随版本变）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.chat import agent_service, tool_registry
from app.utils.api_key_crypto import encrypt_api_key
from app.workspace.local import LocalWorkspace


def _fake_ua() -> SimpleNamespace:
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


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class RegistryShapeTests(unittest.TestCase):
    def test_no_duplicate_specs(self):
        names = [s.name for s in tool_registry.TOOL_SPECS]
        self.assertEqual(len(names), len(set(names)), "TOOL_SPECS 里有重名条目")

    def test_workspace_tools_are_builtin(self):
        """U4 的 4 个工具必须在白名单里——这正是改造前漏掉的。"""
        expected = {
            "workspace_list_dir",
            "workspace_read_file",
            "workspace_write_file",
            "workspace_run_command",
        }
        self.assertTrue(expected <= tool_registry.BUILTIN_TOOL_NAMES)

    def test_mcp_module_shares_the_same_object(self):
        """MCP 侧不能是另一份拷贝——拷贝就会再次漂移。"""
        from app.mcp_client.service import _BUILTIN_TOOL_NAMES

        self.assertIs(_BUILTIN_TOOL_NAMES, tool_registry.BUILTIN_TOOL_NAMES)


class DuplicateDetectionTests(unittest.TestCase):
    def test_find_duplicates(self):
        self.assertEqual(tool_registry.find_duplicates([_Tool("a"), _Tool("b")]), {})
        self.assertEqual(
            tool_registry.find_duplicates([_Tool("a"), _Tool("a"), _Tool("b")]), {"a": 2}
        )

    def test_blank_names_ignored(self):
        """空名不算重名，否则会被误判。"""
        self.assertEqual(tool_registry.find_duplicates([_Tool(""), _Tool("")]), {})

    def test_assert_raises_on_duplicate(self):
        with self.assertRaises(ValueError) as ctx:
            tool_registry.assert_no_duplicates([_Tool("dup"), _Tool("dup")])
        self.assertIn("dup", str(ctx.exception))

    def test_assert_passes_on_unique(self):
        tool_registry.assert_no_duplicates([_Tool("a"), _Tool("b")])


class UnknownToolWarningTests(unittest.TestCase):
    def test_reports_unregistered(self):
        unknown = tool_registry.warn_unknown_tools([_Tool("brand_new_tool")])
        self.assertEqual(unknown, ["brand_new_tool"])

    def test_exempt_covers_dynamic_tools(self):
        """MCP 工具是动态的，通过 exempt 排除，不该被当成遗漏。"""
        unknown = tool_registry.warn_unknown_tools(
            [_Tool("mcp_whatever")], exempt=["mcp_whatever"]
        )
        self.assertEqual(unknown, [])


class ActualRegistrationCoverageTests(unittest.TestCase):
    """实际注册的工具名必须与注册表一致（T1）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = LocalWorkspace(Path(self._tmp.name))
        self.ua = _fake_ua()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _registered_names(self, **kwargs) -> set[str]:
        with mock.patch("langchain.agents.create_agent", return_value=mock.Mock()) as m:
            agent_service.build_model_and_agent(self.ua, 1, 1, "s-coverage", **kwargs)
            tools = m.call_args.kwargs.get("tools") or []
        return {t.name for t in tools}

    def test_all_registered_tools_are_declared(self):
        """含 workspace + exec 的最大集合，必须全部在注册表里。"""
        names = self._registered_names(
            workspace=self.ws,
            checkpointer=mock.Mock(),
            interrupt_on={"workspace_run_command": True},
        )
        undeclared = names - tool_registry.BUILTIN_TOOL_NAMES
        self.assertEqual(undeclared, set(), f"以下工具未在 TOOL_SPECS 登记：{sorted(undeclared)}")

    def test_declared_tools_exist_at_runtime(self):
        """反向：注册表里声明的名字都该真实产出，避免留下幽灵条目。

        `conditional=True` 的工具（目前只有 `search_tools`）默认配置下不产出，
        单独排除——它的产出由 `tests/chat/test_tool_gateway.py` 覆盖。
        """
        names = self._registered_names(
            workspace=self.ws,
            checkpointer=mock.Mock(),
            interrupt_on={"workspace_run_command": True},
        )
        always_on = frozenset(
            s.name for s in tool_registry.TOOL_SPECS if s.builtin and not s.conditional
        )
        missing = always_on - names
        self.assertEqual(
            missing, frozenset(), f"注册表声明了但未实际注册：{sorted(missing)}"
        )

    def test_assembly_raises_on_duplicate(self):
        """重名工具在装配期就该抛错，不能交给 LangChain 静默处理。"""
        dup = _Tool("search_knowledge_base")
        with self.assertRaises(ValueError):
            with mock.patch("langchain.agents.create_agent", return_value=mock.Mock()):
                agent_service.build_model_and_agent(
                    self.ua, 1, 1, "s-dup", extra_tools=[dup]
                )


if __name__ == "__main__":
    unittest.main()
