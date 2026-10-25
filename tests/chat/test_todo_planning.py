"""U9a 规划（todo 列表）：装配开关、updates 流解析、去重、撞名保留。

拆成四组：
1. 装配：开关开/关决定 `TodoListMiddleware` 在不在 middleware 链里
2. 解析：`_extract_todos` 面对真实 payload / 脏数据 / 无关 payload 的行为
3. 去重：同一份清单重复写入不重复推事件
4. 撞名：`write_todos` 必须在 MCP 保留名里，否则 MCP server 提供同名工具会双绑

真实 payload 形状是实测出来的（2026-10-24，langchain 1.2.15）：
    {"tools": {"todos": [...], "messages": [...]}}
`write_todos` 返回 `Command(update={"todos": ...})`，该更新归到 tools 节点。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase, mock

from app.chat import agent_service
from app.chat.tool_registry import RESERVED_TOOL_NAMES
from app.settings import settings
from app.utils.api_key_crypto import encrypt_api_key


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


class TodoAssemblyTests(TestCase):
    """装配期：开关决定模型手里有没有 write_todos。"""

    def _middlewares(self, enabled: bool) -> list:
        with mock.patch.object(settings, "CHAT_TODO_ENABLED", enabled):
            with mock.patch("langchain.agents.create_agent", return_value=mock.Mock()) as m:
                agent_service.build_model_and_agent(_fake_ua(), 1, 1, "s1")
        return m.call_args.kwargs.get("middleware") or []

    def _middleware_names(self, enabled: bool) -> list[str]:
        return [type(mw).__name__ for mw in self._middlewares(enabled)]

    def test_switch_off_has_no_planner(self):
        self.assertNotIn("TodoListMiddleware", self._middleware_names(False))

    def test_switch_on_adds_planner(self):
        self.assertIn("TodoListMiddleware", self._middleware_names(True))

    def test_planner_prompt_is_imperative(self):
        """线上翻车复盘点：上游提示词写得太软，deepseek 面对明确多步任务也不调用。

        这里守住「我们注入的是自己的硬条件提示词」，而不是上游那句
        "use it when it is helpful"。改回默认值会让 bug 复现。
        """
        planner = next(
            mw for mw in self._middlewares(True) if type(mw).__name__ == "TodoListMiddleware"
        )
        self.assertEqual(planner.system_prompt, agent_service._TODO_SYSTEM_PROMPT)
        # 硬触发条件必须在（3 步 / 2 个工具 / 多件事）
        self.assertIn("必须", planner.system_prompt)
        self.assertIn("3 个及以上步骤", planner.system_prompt)
        self.assertIn("2 个及以上工具", planner.system_prompt)
        # 状态纪律
        self.assertIn("in_progress", planner.system_prompt)
        self.assertIn("completed", planner.system_prompt)
        # 工具描述也要跟着改，否则「可选」的措辞会与系统提示词打架
        self.assertIn("必须先调用本工具", planner.tool_description)

    def test_planner_survives_approval_failure(self):
        """审批中间件装配失败时会把 middlewares 整体重置——规划不能被一起抹掉。

        这是 `build_model_and_agent` 里既有的坑：HITL 的 except 分支写的是
        `middlewares = [strip_mw]`。规划必须追加在它之后，否则开关打开也白开。
        """
        with mock.patch.object(settings, "CHAT_TODO_ENABLED", True):
            with mock.patch.object(
                agent_service, "_approval_tools_map", return_value={"workspace_run_command": True}
            ):
                # patch 的必须是**生产代码真正用的那个名字**（app.chat.approval 里的
                # SafeCommandAwareHITL）。原先 patch 的是 langchain 模块里的类名，
                # 在 U7-只读白名单把装配改成子类后就打空了；更糟的是它会污染
                # 首次导入：如果 app.chat.approval 恰好在此 patch 期间被导入，
                # 类会继承到 Mock —— 单跑通过、全量跑报 StopIteration。
                with mock.patch(
                    "app.chat.approval.SafeCommandAwareHITL",
                    side_effect=RuntimeError("boom"),
                ):
                    with mock.patch(
                        "langchain.agents.create_agent", return_value=mock.Mock()
                    ) as m:
                        agent_service.build_model_and_agent(
                            _fake_ua(), 1, 1, "s1", checkpointer=object(), interrupt_on={"x": True}
                        )
        names = [type(mw).__name__ for mw in (m.call_args.kwargs.get("middleware") or [])]
        self.assertIn("TodoListMiddleware", names)


class ExtractTodosTests(TestCase):
    """updates payload → 前端契约结构。"""

    def test_real_payload_shape(self):
        payload = {
            "tools": {
                "todos": [
                    {"content": "查知识库", "status": "in_progress"},
                    {"content": "写总结", "status": "pending"},
                ],
                "messages": ["...忽略..."],
            }
        }
        self.assertEqual(
            agent_service._extract_todos(payload),
            [
                {"content": "查知识库", "status": "in_progress"},
                {"content": "写总结", "status": "pending"},
            ],
        )

    def test_unknown_status_falls_back_to_pending(self):
        payload = {"tools": {"todos": [{"content": "步骤", "status": "DONE!"}]}}
        self.assertEqual(agent_service._extract_todos(payload), [{"content": "步骤", "status": "pending"}])

    def test_blank_content_is_dropped(self):
        payload = {"tools": {"todos": [{"content": "   ", "status": "pending"}, {"content": "有效"}]}}
        self.assertEqual(agent_service._extract_todos(payload), [{"content": "有效", "status": "pending"}])

    def test_empty_list_is_not_an_update(self):
        """空清单当作「没有更新」——否则会把前端已显示的进度清空。"""
        self.assertIsNone(agent_service._extract_todos({"tools": {"todos": []}}))

    def test_unrelated_payloads_are_ignored(self):
        self.assertIsNone(agent_service._extract_todos({"__interrupt__": (object(),)}))
        self.assertIsNone(agent_service._extract_todos({"model": {"messages": []}}))
        self.assertIsNone(agent_service._extract_todos(None))
        self.assertIsNone(agent_service._extract_todos("not-a-dict"))


class TodoCollectorTests(TestCase):
    """去重：模型在同一步反复写同样的表，不该让前端白重渲染。"""

    def _collector(self):
        return agent_service._ThinkingCollector()

    def test_same_list_is_not_pushed_twice(self):
        c = self._collector()
        todos = [{"content": "a", "status": "pending"}]
        c.set_todos(todos)
        c.set_todos(list(todos))
        events = c.drain()
        self.assertEqual([e["type"] for e in events], ["todo_list"])
        self.assertEqual(events[0]["todos"], todos)

    def test_changed_list_is_pushed_again(self):
        c = self._collector()
        c.set_todos([{"content": "a", "status": "pending"}])
        c.set_todos([{"content": "a", "status": "completed"}])
        self.assertEqual(len([e for e in c.drain() if e["type"] == "todo_list"]), 2)

    def test_empty_update_is_noop(self):
        c = self._collector()
        c.set_todos(None)
        c.set_todos([])
        self.assertEqual(c.drain(), [])
        self.assertIsNone(c.latest_todos)

    def test_latest_todos_tracks_last_push(self):
        c = self._collector()
        c.set_todos([{"content": "a", "status": "pending"}])
        c.set_todos([{"content": "b", "status": "completed"}])
        self.assertEqual(c.latest_todos, [{"content": "b", "status": "completed"}])


class RecursionLimitTests(TestCase):
    """步数上限：曾经写死 30，多步任务必然打穿（线上实录）。"""

    def test_default_is_raised(self):
        from app.chat.agent_prompt import _recursion_limit

        with mock.patch.object(settings, "CHAT_RECURSION_LIMIT", 60):
            self.assertEqual(_recursion_limit(), 60)

    def test_settings_override_is_used(self):
        from app.chat.agent_prompt import _recursion_limit

        with mock.patch.object(settings, "CHAT_RECURSION_LIMIT", 120):
            self.assertEqual(_recursion_limit(), 120)

    def test_floor_prevents_instant_failure(self):
        """配成 0/负数会让每轮立刻失败，比放宽更糟——下限兜到 10。"""
        from app.chat.agent_prompt import _recursion_limit

        for bad in (0, -5):
            with mock.patch.object(settings, "CHAT_RECURSION_LIMIT", bad):
                self.assertEqual(_recursion_limit(), 10)

    def test_invoke_config_carries_limit(self):
        from app.chat.agent_prompt import _agent_invoke_config

        with mock.patch.object(settings, "CHAT_RECURSION_LIMIT", 88):
            cfg = _agent_invoke_config()
        self.assertEqual(cfg["recursion_limit"], 88)


class ReservedNameTests(TestCase):
    """撞名：write_todos 由中间件注入，MCP 过滤必须认得它。"""

    def test_write_todos_is_reserved(self):
        self.assertIn("write_todos", RESERVED_TOOL_NAMES)

    def test_reserved_not_in_builtin_specs(self):
        """它不该出现在 TOOL_SPECS 里——那是 provider 产的清单，中间件工具不属于那里。"""
        from app.chat.tool_registry import BUILTIN_TOOL_NAMES

        self.assertNotIn("write_todos", BUILTIN_TOOL_NAMES)

    def test_mcp_filter_uses_reserved_names(self):
        """MCP 加载时的 seen_names 必须包含保留名，否则同名工具会双绑到同一个模型。"""
        import inspect

        import app.mcp_client.service as svc

        src = inspect.getsource(svc.load_agent_mcp_tools)
        self.assertIn("_RESERVED_TOOL_NAMES", src)


if __name__ == "__main__":
    import unittest

    unittest.main()
