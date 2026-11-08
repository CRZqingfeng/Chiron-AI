"""工作区与 agent 装配的接线测试。

守住三个容易在后续改动中悄悄失守的约束：
1. 传了 workspace 才注册工作区工具，不传则工具数与改造前一致
2. 只暴露三个文件类工具，`execute` 不得出现在工具列表里（U5b 前不能有）
3. 工具名统一带 `workspace_` 前缀

做法是拦截 `langchain.agents.create_agent` 抓它实际收到的 tools——
比去翻 CompiledStateGraph 的内部结构可靠得多（后者是私有实现，会随版本变）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.chat import agent_service
from app.utils.api_key_crypto import encrypt_api_key
from app.workspace.local import LocalWorkspace

WORKSPACE_TOOLS = {"workspace_list_dir", "workspace_read_file", "workspace_write_file"}


def _fake_ua() -> SimpleNamespace:
    """构造够 `build_model_and_agent` 跑通的最小 ua。

    Key 用真实加密函数生成——`decrypt_api_key_safe` 会在解密失败时返回 None，
    进而抛「未配置有效 API Key」，那样就测不到工具装配了。
    """
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


class WorkspaceWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = LocalWorkspace(Path(self._tmp.name))
        self.ua = _fake_ua()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _captured_tools(self, **kwargs) -> list[str]:
        with mock.patch("langchain.agents.create_agent", return_value=mock.Mock()) as m:
            agent_service.build_model_and_agent(self.ua, 9902, 9901, "s1", **kwargs)
            tools = m.call_args.kwargs.get("tools") or []
        return sorted(t.name for t in tools)

    def test_workspace_tools_registered_when_passed(self):
        names = set(self._captured_tools(workspace=self.ws))
        self.assertTrue(WORKSPACE_TOOLS.issubset(names), sorted(WORKSPACE_TOOLS - names))

    def test_no_workspace_tools_when_omitted(self):
        names = set(self._captured_tools())
        self.assertEqual(names & WORKSPACE_TOOLS, set())

    def test_workspace_adds_exactly_six_tools(self):
        """U10 P0 之后是六个：list / read / write / edit / grep / glob。"""
        with_ws = self._captured_tools(workspace=self.ws)
        without = self._captured_tools()
        self.assertEqual(len(with_ws) - len(without), 6)

    def test_execute_is_not_exposed(self):
        """U5b（审批中断）落地之前，模型不得具备执行任意 shell 的能力。

        谁哪天顺手把 `execute` 包成 tool 加进来，这条会红。
        """
        names = self._captured_tools(workspace=self.ws)
        for name in names:
            self.assertNotIn("exec", name)
            self.assertNotIn("run_command", name)
            self.assertNotIn("shell", name)

    def test_existing_tool_count_unchanged_without_workspace(self):
        """不加工作区时，工具集合必须与改造前完全一致——这是回归的主要风险点。"""
        names = set(self._captured_tools())
        expected = {
            "search_session_attachment",
            "read_session_attachment",
            "list_session_attachments_brief",
        }
        self.assertTrue(expected.issubset(names), sorted(expected - names))
        self.assertEqual(len(names & WORKSPACE_TOOLS), 0)


if __name__ == "__main__":
    unittest.main()
