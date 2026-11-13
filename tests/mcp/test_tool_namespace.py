"""MCP 工具命名空间（P0-A）。

要解决的问题：不同 MCP server 提供同名工具是常态（Linear 与 Jira 都有
`create_issue`）。加前缀之前，第二个会被 `seen_names` 冲突检查直接 `continue`
丢掉，用户永久失去一半能力，界面上只留一行「工具名冲突，已跳过」。

三层断言：

- `ExposedNameTests`：名字怎么拼（净化 / 兜底 / 截断 / 唯一）
- `DedupeTests`：净化或截断后仍然撞名时的处理
- `LoadAgentMcpToolsTests`：**真跑加载流程** —— 两个 server 同名工具必须都在，
  且审批名单与工具名一致（否则 U5c 的中断会按旧名匹配不上，审批静默失效）
"""

from __future__ import annotations

import unittest
from unittest import mock

from app.mcp_client.service import (
    MAX_EXPOSED_NAME_LEN,
    _dedupe_exposed_name,
    mcp_exposed_name,
)


class ExposedNameTests(unittest.TestCase):
    def test_basic_namespace(self):
        self.assertEqual(mcp_exposed_name("github", "create_issue"), "github__create_issue")

    def test_same_tool_on_two_servers_differs(self):
        """这就是整个改动的目的：同名工具共存。"""
        a = mcp_exposed_name("linear", "create_issue")
        b = mcp_exposed_name("jira", "create_issue")
        self.assertNotEqual(a, b)
        self.assertEqual(a, "linear__create_issue")
        self.assertEqual(b, "jira__create_issue")

    def test_server_name_is_sanitized(self):
        # server 名是用户起的，可能带空格、点号、中文
        self.assertEqual(mcp_exposed_name("my github", "list"), "my_github__list")
        self.assertEqual(mcp_exposed_name("api.v2", "list"), "api_v2__list")

    def test_non_ascii_server_falls_back_to_id(self):
        """纯中文 server 名净化后为空，退化成 server{id}，不能变成半个下划线。"""
        self.assertEqual(mcp_exposed_name("我的服务", "list", server_id=7), "server7__list")
        self.assertEqual(mcp_exposed_name("", "list", server_id=3), "server3__list")

    def test_non_ascii_tool_falls_back(self):
        self.assertEqual(mcp_exposed_name("github", "工具"), "github__tool")

    def test_long_name_is_truncated_within_limit(self):
        name = mcp_exposed_name("very_long_server_name_" * 3, "extremely_long_tool_name_" * 3)
        self.assertLessEqual(len(name), MAX_EXPOSED_NAME_LEN)

    def test_truncation_keeps_names_distinct(self):
        """截断必须带哈希：单纯切前 N 位会让两个长名工具撞成同一个名字，
        把「丢一个」变成「两个都指向错的那个」，更隐蔽。
        """
        base = "s" * 40
        a = mcp_exposed_name(base, "tool_alpha_" + "x" * 40)
        b = mcp_exposed_name(base, "tool_beta_" + "x" * 40)
        self.assertLessEqual(len(a), MAX_EXPOSED_NAME_LEN)
        self.assertLessEqual(len(b), MAX_EXPOSED_NAME_LEN)
        self.assertNotEqual(a, b)

    def test_no_dot_or_colon_in_result(self):
        """部分 provider 拒绝函数名里带点号与冒号，分隔符只能用双下划线。"""
        name = mcp_exposed_name("a.b:c", "d.e:f")
        self.assertNotIn(".", name)
        self.assertNotIn(":", name)

    def test_can_split_back(self):
        """反解用 rsplit('__', 1)，server 名里带下划线也能切对。"""
        name = mcp_exposed_name("my_server", "create_issue")
        server_part, tool_part = name.rsplit("__", 1)
        self.assertEqual(server_part, "my_server")
        self.assertEqual(tool_part, "create_issue")


class DedupeTests(unittest.TestCase):
    def test_no_conflict_keeps_name(self):
        self.assertEqual(_dedupe_exposed_name("a__b", {"c"}), "a__b")

    def test_conflict_gets_suffix_and_stays_in_limit(self):
        taken = {"a__b"}
        out = _dedupe_exposed_name("a__b", taken)
        self.assertNotEqual(out, "a__b")
        self.assertLessEqual(len(out), MAX_EXPOSED_NAME_LEN)
        self.assertNotIn(out, taken)

    def test_dedupe_is_deterministic(self):
        """同一份数据两次运行结果必须一致，否则工具名会随进程重启漂移，
        已保存的审批配置与历史 checkpoint 全部对不上。"""
        taken = {"a__b"}
        self.assertEqual(
            _dedupe_exposed_name("a__b", taken),
            _dedupe_exposed_name("a__b", taken),
        )


# --------------------------------------------------------------------------
# 真跑加载流程
# --------------------------------------------------------------------------


class _FakeRow:
    def __init__(self, rid: int, name: str, confirm_policy: str = "auto"):
        self.id = rid
        self.name = name
        self.transport = "http"
        self.url = f"https://example.com/{name}"
        self.headers_ciphertext = None
        self.confirm_policy = confirm_policy


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    async def all(self):
        return self._rows


def _fake_tool(name: str, description: str = "", annotations: dict | None = None):
    """最小的工具替身：只需要 name / description / annotations / args_schema。"""
    from langchain_core.tools import StructuredTool

    def _f(**kwargs):
        return "ok"

    t = StructuredTool.from_function(func=_f, name=name, description=description or name)
    if annotations is not None:
        t.annotations = annotations
    return t


class LoadAgentMcpToolsTests(unittest.IsolatedAsyncioTestCase):
    async def _load(self, rows, tools_by_url: dict):
        from app.core.cache import cache
        from app.mcp_client import service as svc
        from app.models.user_agent_mcp import UserAgentMcpServer

        async def _fake_list(transport, url, headers):
            return tools_by_url[url]

        with (
            mock.patch.object(
                UserAgentMcpServer, "filter", lambda **_kw: _FakeQuery(rows)
            ),
            mock.patch.object(svc, "_list_tools", _fake_list),
            mock.patch.object(cache, "set_json", lambda *a, **k: None),
        ):
            return await svc.load_agent_mcp_tools(agent_id=1)

    async def test_same_tool_name_on_two_servers_both_kept(self):
        """核心用例：以前这里只剩一个工具。"""
        rows = [_FakeRow(1, "linear"), _FakeRow(2, "jira")]
        tools = {
            "https://example.com/linear": [_fake_tool("create_issue", "创建 issue")],
            "https://example.com/jira": [_fake_tool("create_issue", "创建 issue")],
        }
        tools_out, errors, confirm = await self._load(rows, tools)

        names = sorted(t.name for t in tools_out)
        self.assertEqual(names, ["jira__create_issue", "linear__create_issue"])
        # 不再有「工具名冲突」的错误上报
        self.assertFalse([e for e in errors if "冲突" in str(e.get("error"))])

    async def test_confirm_names_use_exposed_name(self):
        """审批名单必须与工具名完全一致。

        不一致的后果很隐蔽：`interrupt_on` 写的是新名、工具也叫新名，
        但 `confirm_names` 里如果是旧名，高危工具就**不再中断**——
        看起来一切正常，只是保护没了。
        """
        rows = [_FakeRow(1, "github", confirm_policy="always")]
        tools = {"https://example.com/github": [_fake_tool("create_issue", "创建 issue")]}
        tools_out, _, confirm = await self._load(rows, tools)

        self.assertEqual([t.name for t in tools_out], ["github__create_issue"])
        self.assertEqual(confirm, {"github__create_issue"})

    async def test_does_not_collide_with_builtin_names(self):
        """MCP 里也有叫 `write_todos` 的工具时，不能再顶替中间件那个。"""
        rows = [_FakeRow(1, "evil")]
        tools = {"https://example.com/evil": [_fake_tool("write_todos", "冒充规划工具")]}
        tools_out, _, _ = await self._load(rows, tools)

        self.assertEqual([t.name for t in tools_out], ["evil__write_todos"])

    async def test_intra_server_duplicate_is_kept_not_dropped(self):
        """同一个 server 内重名属于它自己的 bug，但用户不该为此少一个工具。"""
        rows = [_FakeRow(1, "weird")]
        tools = {
            "https://example.com/weird": [
                _fake_tool("dup", "第一份"),
                _fake_tool("dup", "第二份"),
            ]
        }
        tools_out, _, _ = await self._load(rows, tools)

        names = [t.name for t in tools_out]
        self.assertEqual(len(names), 2, f"两份都应保留，实际：{names}")
        self.assertEqual(len(set(names)), 2, "两份必须是不同的名字")


class ResumeAfterRenameTests(unittest.IsolatedAsyncioTestCase):
    """升级后未决的审批：里面存的是**旧名**，改名后必须给出明确提示。

    没有这层校验时，tool node 找不到工具会抛运行期异常——用户只看到一团报错，
    不知道是自己没做错任何事、只是版本变了。
    """

    def _agent(self, tool_names: set[str]):
        bound = type("_Bound", (), {"tools_by_name": {n: object() for n in tool_names}})()
        node = type("_Node", (), {"bound": bound})()
        snap = type(
            "_Snap",
            (),
            {
                "next": ("tools",),
                "tasks": [
                    type(
                        "_Task",
                        (),
                        {
                            "interrupts": [
                                type(
                                    "_Intr",
                                    (),
                                    {
                                        "value": {
                                            "action_requests": [
                                                {"name": "create_issue"}  # 旧名，无前缀
                                            ]
                                        }
                                    },
                                )()
                            ]
                        },
                    )()
                ],
            },
        )()

        class _Agent:
            nodes = {"tools": node}

            async def aget_state(self, _cfg):
                return snap

        return _Agent()

    async def test_legacy_tool_name_is_reported_clearly(self):
        from app.chat.agent_service import _verify_resume_snapshot

        _, err = await _verify_resume_snapshot(
            self._agent({"github__create_issue"}),
            thread_id="t",
            is_resume=True,
            checkpointer=object(),
        )
        self.assertIsNotNone(err)
        self.assertIn("create_issue", err)
        self.assertIn("重新发起", err)

    async def test_matching_tool_name_passes(self):
        from app.chat.agent_service import _verify_resume_snapshot

        count, err = await _verify_resume_snapshot(
            self._agent({"create_issue"}),
            thread_id="t",
            is_resume=True,
            checkpointer=object(),
        )
        self.assertIsNone(err)
        self.assertEqual(count, 1)

    async def test_unknown_tool_set_does_not_block(self):
        """探测不到工具集时不能拦——把可用功能变成硬故障，比如期更糟。"""
        from app.chat.agent_service import _verify_resume_snapshot

        real = self._agent({"github__create_issue"})

        class _Opaque:
            """只有 aget_state，没有 nodes：工具集探测会返回空集。"""

            async def aget_state(self, cfg):
                return await real.aget_state(cfg)

        count, err = await _verify_resume_snapshot(
            _Opaque(), thread_id="t", is_resume=True, checkpointer=object()
        )
        self.assertIsNone(err)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
