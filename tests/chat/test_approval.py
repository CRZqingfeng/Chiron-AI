"""审批中断的后端单元测试（U5a-min）。

不依赖 Docker、PG、LLM：
- `_approval_tools_map` / `_interrupt_to_event` 是纯函数
- checkpointer 降级用 mock 触发
- 工具注册面用 LocalWorkspace
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.chat import agent_service
from app.chat import checkpoint as ckpt
from app.settings import settings
from app.workspace.local import LocalWorkspace
from app.workspace.tools import make_workspace_tools


class ApprovalToolsMapTests(unittest.TestCase):
    def test_parses_comma_separated(self):
        with mock.patch.object(settings, "CHAT_APPROVAL_TOOLS", "a, b ,,c"):
            self.assertEqual(agent_service._approval_tools_map(), {"a": True, "b": True, "c": True})

    def test_empty_means_nothing_needs_approval(self):
        with mock.patch.object(settings, "CHAT_APPROVAL_TOOLS", "  "):
            self.assertEqual(agent_service._approval_tools_map(), {})


class InterruptToEventTests(unittest.TestCase):
    def _interrupt(self):
        value = {
            "action_requests": [
                {"name": "workspace_run_command", "args": {"command": "ls"}, "description": "确认执行"}
            ],
            "review_configs": [
                {"action_name": "workspace_run_command", "allowed_decisions": ["approve", "edit", "reject"]}
            ],
        }
        return SimpleNamespace(value=value, id="int-1")

    def test_flattens_single_interrupt(self):
        out = agent_service._interrupt_to_event((self._interrupt(),))
        self.assertEqual(out["interrupt_id"], "int-1")
        self.assertEqual(len(out["actions"]), 1)
        self.assertEqual(out["actions"][0]["name"], "workspace_run_command")
        self.assertEqual(out["actions"][0]["args"], {"command": "ls"})
        self.assertEqual(out["allowed_decisions"], ["approve", "edit", "reject"])

    def test_defaults_decisions_when_absent(self):
        """取不到 allowed_decisions 时退化成 approve/reject，不能返回空列表——
        空列表会让前端渲染出一个无法操作的卡片。"""
        it = SimpleNamespace(value={"action_requests": [{"name": "x", "args": {}}]}, id="i2")
        out = agent_service._interrupt_to_event((it,))
        self.assertEqual(out["allowed_decisions"], ["approve", "reject"])

    def test_handles_plain_dict(self):
        out = agent_service._interrupt_to_event({"action_requests": [], "review_configs": []})
        self.assertEqual(out["actions"], [])
        self.assertIsNone(out["interrupt_id"])


class ExecToolGatingTests(unittest.TestCase):
    """`run_command` 必须与审批同时到位：默认不注册。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = LocalWorkspace(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_exec_not_registered_by_default(self):
        names = {t.name for t in make_workspace_tools(self.ws)}
        self.assertNotIn("workspace_run_command", names)
        self.assertEqual(
            names,
            {
                "workspace_edit_file",
                "workspace_list_dir",
                "workspace_read_file",
                "workspace_write_file",
            },
        )

    def test_exec_registered_when_explicitly_allowed(self):
        names = {t.name for t in make_workspace_tools(self.ws, include_exec=True)}
        self.assertIn("workspace_run_command", names)
        self.assertEqual(len(names), 5)

    def test_exec_tool_returns_formatted_result(self):
        tool = next(t for t in make_workspace_tools(self.ws, include_exec=True) if t.name == "workspace_run_command")
        out = tool.invoke({"command": "echo hi-approval"})
        self.assertIn("退出码：0", out)
        self.assertIn("hi-approval", out)

    def test_exec_tool_reports_nonzero(self):
        tool = next(t for t in make_workspace_tools(self.ws, include_exec=True) if t.name == "workspace_run_command")
        out = tool.invoke({"command": "exit 4"})
        self.assertIn("退出码：4", out)


class CheckpointerFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        asyncio.run(ckpt.reset_for_tests())

    def tearDown(self) -> None:
        asyncio.run(ckpt.reset_for_tests())

    def test_disabled_returns_none(self):
        with mock.patch.object(settings, "CHAT_APPROVAL_ENABLED", False), mock.patch.object(
            ckpt, "_enabled", return_value=False
        ):
            self.assertIsNone(asyncio.run(ckpt.get_checkpointer()))

    def test_backend_none_returns_none(self):
        with mock.patch.object(ckpt, "_backend", return_value="none"):
            self.assertIsNone(asyncio.run(ckpt.get_checkpointer()))

    def test_postgres_failure_falls_back_to_memory(self):
        """PG 不可用时必须降级而不是抛异常——对话不能因为审批存储失败而中断。"""
        with mock.patch.object(ckpt, "_backend", return_value="postgres"), mock.patch.object(
            ckpt, "_build_postgres", side_effect=RuntimeError("pg down")
        ):
            saver = asyncio.run(ckpt.get_checkpointer())
        self.assertIsNotNone(saver)
        self.assertEqual(type(saver).__name__, "InMemorySaver")

    def test_memory_backend_returns_in_memory(self):
        with mock.patch.object(ckpt, "_backend", return_value="memory"):
            saver = asyncio.run(ckpt.get_checkpointer())
        self.assertEqual(type(saver).__name__, "InMemorySaver")

    def test_conninfo_strips_sqlalchemy_driver(self):
        """psycopg 只认裸 scheme，`+psycopg` 会让它连不上。"""
        self.assertEqual(
            ckpt._to_conninfo("postgresql+psycopg://u:p@h:5/db"),
            "postgresql://u:p@h:5/db",
        )

    def test_conninfo_leaves_plain_url_untouched(self):
        self.assertEqual(ckpt._to_conninfo("postgresql://u:p@h/db"), "postgresql://u:p@h/db")


if __name__ == "__main__":
    unittest.main()


class CheckpointCleanupTests(unittest.TestCase):
    """checkpoint 过期清理的时间解析与短路条件。"""

    def test_uuid6_time_roundtrip(self):
        """能按给定时间合成 uuid6，再反解回来（误差 < 1 秒）。"""
        import time

        ts = time.time() - 3 * 86400
        hundred = int((ts + 12219292800) * 1e7)
        raw = (
            f"{(hundred >> 28) & 0xFFFFFFFF:08x}-{(hundred >> 12) & 0xFFFF:04x}"
            f"-6{hundred & 0x0FFF:03x}-8000-000000000000"
        )
        parsed = ckpt._uuid6_unix_time(raw)
        self.assertIsNotNone(parsed)
        self.assertLess(abs(parsed - ts), 1.0)

    def test_non_v6_returns_none(self):
        """不是 v6 就返回 None —— 清理逻辑据此「不删」，避免误删未知格式的线程。"""
        self.assertIsNone(ckpt._uuid6_unix_time("not-a-uuid"))
        # v4 是随机的，不含时间
        self.assertIsNone(ckpt._uuid6_unix_time("3f4a1b2c-9d8e-4f6a-b7c5-1e2d3c4b5a69"))

    def test_purge_returns_zero_without_pool(self):
        """池没建起来时直接返回 0，不抛异常。"""
        with mock.patch.object(ckpt, "_pool", None):
            self.assertEqual(asyncio.run(ckpt.purge_expired_threads(1)), 0)

    def test_purge_zero_days_is_noop(self):
        with mock.patch.object(ckpt, "_pool", object()):
            self.assertEqual(asyncio.run(ckpt.purge_expired_threads(0)), 0)


class StorageApprovalFieldTests(unittest.TestCase):
    """`approval_pending` 必须能落库并能读回——否则中断后刷新页面卡片就丢了。"""

    def test_parse_extra_keeps_approval_pending(self):
        from app.chat.storage import ConversationStorage

        fields = ConversationStorage._parse_extra(  # type: ignore[attr-defined]
            {"approval_pending": {"thread_id": "j1", "actions": []}}
        )
        self.assertEqual(fields["approval_pending"], {"thread_id": "j1", "actions": []})

    def test_parse_extra_without_it_is_none(self):
        from app.chat.storage import ConversationStorage

        fields = ConversationStorage._parse_extra({})  # type: ignore[attr-defined]
        self.assertIsNone(fields["approval_pending"])


class ResumeRequestValidationTests(unittest.TestCase):
    """`resume_decision` 的入口校验（安全与体验双重要求）。"""

    def test_accepts_approve_and_reject(self):
        from app.schemas.agent_chat import ChatRequest

        for v in ("approve", "reject", "APPROVE", " reject "):
            with self.subTest(v=v):
                req = ChatRequest(agent_id=1, message="hi", resume_decision=v)
                self.assertIn(req.resume_decision, ("approve", "reject"))

    def test_rejects_unknown_decision(self):
        """非法值必须在入口拦住。

        放过去会一路到 HITL middleware 抛 ValueError，被错误分类器归为 unknown，
        用户只看到「生成失败」；更糟的是那个 thread 会留下无法恢复的状态。
        """
        from pydantic import ValidationError

        from app.schemas.agent_chat import ChatRequest

        for bad in ("whatever", "yes", "approve_all"):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                ChatRequest(agent_id=1, message="hi", resume_decision=bad)

    def test_blank_decision_becomes_none(self):
        from app.schemas.agent_chat import ChatRequest

        self.assertIsNone(ChatRequest(agent_id=1, message="hi", resume_decision="   ").resume_decision)


class ResumeOwnershipContractTests(unittest.TestCase):
    """属主校验的契约（实现在 `iter_chat_stream_events`，此处守住判定条件本身）。

    越权是实测确认过的真实漏洞：只按 `resume_thread_id` 去 recover checkpoint 时，
    换个 user_id / session_id 也能恢复别人的会话并触发对方被中断的命令执行。
    """

    def test_thread_must_match_message_approval_pending(self):
        """判定条件：目标消息的 `approval_pending.thread_id` 必须等于请求里的 thread。"""

        def allowed(record: dict | None, resume_thread_id: str) -> bool:
            pending = (record or {}).get("approval_pending") or {}
            return bool(record) and bool(pending) and pending.get("thread_id") == resume_thread_id

        self.assertTrue(allowed({"approval_pending": {"thread_id": "t1"}}, "t1"))
        # 消息不属于本会话（查不到）→ 拒
        self.assertFalse(allowed(None, "t1"))
        # 消息是普通助手消息、没有待确认信息 → 拒
        self.assertFalse(allowed({"approval_pending": None}, "t1"))
        # thread 对不上（伪造或跨会话）→ 拒
        self.assertFalse(allowed({"approval_pending": {"thread_id": "t1"}}, "t2"))


class CountPendingActionsTests(unittest.TestCase):
    """resume 时 `decisions` 的填充数量（U5b）。

    HITL middleware 强制 `len(decisions) == len(interrupted_tool_calls)`，
    数量不足抛 ValueError；所以这个计数错了，多工具中断就恢复不了。
    """

    @staticmethod
    def _snapshot(actions_per_interrupt: list[int]):
        tasks = []
        for n in actions_per_interrupt:
            intr = SimpleNamespace(
                value={"action_requests": [{"name": f"t{i}"} for i in range(n)]}
            )
            tasks.append(SimpleNamespace(interrupts=(intr,)))
        return SimpleNamespace(tasks=tuple(tasks))

    def test_single_action(self):
        self.assertEqual(agent_service._count_pending_actions(self._snapshot([1])), 1)

    def test_multiple_actions(self):
        """模型一次调 2 个需审批工具——这正是 U5a-full 漏掉的场景。"""
        self.assertEqual(agent_service._count_pending_actions(self._snapshot([2])), 2)
        self.assertEqual(agent_service._count_pending_actions(self._snapshot([3])), 3)

    def test_empty_snapshot_falls_back_to_one(self):
        """数不出来时退回 1：保守，不会让原本能跑的单工具场景变坏。"""
        self.assertEqual(agent_service._count_pending_actions(SimpleNamespace(tasks=())), 1)
        self.assertEqual(agent_service._count_pending_actions(SimpleNamespace()), 1)

    def test_non_dict_interrupt_value_is_skipped(self):
        snap = SimpleNamespace(
            tasks=(SimpleNamespace(interrupts=(SimpleNamespace(value="not-a-dict"),)),)
        )
        self.assertEqual(agent_service._count_pending_actions(snap), 1)

    def test_zero_actions_falls_back_to_one(self):
        """action_requests 为空不是「0 个决定」，而是数据异常 → 按 1 处理。"""
        self.assertEqual(agent_service._count_pending_actions(self._snapshot([0])), 1)


class MessageInfoApprovalPassThroughTests(unittest.TestCase):
    """`approval_pending` 必须过得了 `MessageInfo`（U5b）。

    pydantic 默认丢弃未声明字段：U5a-full 只改了 storage，schema 没加，
    于是库里明明有值，接口返回时被静默丢掉——刷新后卡片就消失了。
    """

    @staticmethod
    def _raw(pending):
        row = {
            "message_id": 7,
            "type": "ai",
            "content": "",
            "timestamp": "2026-10-24 12:00:00",
        }
        if pending is not None:
            row["approval_pending"] = pending
        return [row]

    def test_field_survives_schema(self):
        from app.api.v1.user_agents.chat import _to_message_infos

        pending = {
            "thread_id": "job-1",
            "interrupt_id": "intr-1",
            "actions": [{"name": "workspace_run_command", "args": {"command": "ls"}}],
            "allowed_decisions": ["approve", "reject"],
        }
        out = _to_message_infos(self._raw(pending))[0]
        self.assertIsNotNone(out.approval_pending)
        self.assertEqual(out.approval_pending["thread_id"], "job-1")
        self.assertEqual(len(out.approval_pending["actions"]), 1)

    def test_absent_field_stays_none(self):
        from app.api.v1.user_agents.chat import _to_message_infos

        self.assertIsNone(_to_message_infos(self._raw(None))[0].approval_pending)


class ResumeEmptyMessageTests(unittest.TestCase):
    """审批恢复请求允许 `message` 为空（U5b 人工验证暴露）。

    `ChatRequest` 原有校验「message 与 attachment_ids 至少填写一项」只放行了
    `regenerate`。恢复请求两者都没有（模型输入来自 checkpoint），于是在入口被
    422 拦掉——函数本身能跑，用户却发不出去。

    为什么放在这里而不是打 HTTP：`/chat/jobs` 走完中间件需要 Tortoise / Redis
    等运行时上下文，为验一条字段规则引入那些依赖不划算。路由层的透传由
    `create_chat_job(...)` 调用处肉眼可核（resume_thread_id / resume_decision 都在）。
    """

    def test_empty_message_allowed_for_resume(self):
        from app.schemas.agent_chat import ChatRequest

        req = ChatRequest(
            agent_id=1,
            message="",
            resume_decision="approve",
            resume_thread_id="thread-1",
            target_message_id=42,
        )
        self.assertEqual(req.message, "")
        self.assertEqual(req.resume_thread_id, "thread-1")
        self.assertEqual(req.target_message_id, 42)

    def test_empty_message_still_rejected_without_resume(self):
        """放宽不能过头：普通提问仍然要求有内容。"""
        from pydantic import ValidationError

        from app.schemas.agent_chat import ChatRequest

        with self.assertRaises(ValidationError):
            ChatRequest(agent_id=1, message="", attachment_ids=[])

    def test_regenerate_empty_message_still_allowed(self):
        """既有能力不受影响。"""
        from app.schemas.agent_chat import ChatRequest

        self.assertTrue(ChatRequest(agent_id=1, message="", regenerate=True).regenerate)
