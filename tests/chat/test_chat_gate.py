"""并发闸门：排队事件、等待超时置失败。不依赖真实 Redis/上游模型。"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

from tests.chat.test_sse_events import _FakeRedis


def _events(fake: _FakeRedis, job_id: str) -> list[dict]:
    from app.chat import chat_job

    key = f"chiron_ai:{chat_job._events_key(job_id)}"
    return [json.loads(w)["data"] for w in fake.lists.get(key, [])]


def _meta(fake: _FakeRedis, job_id: str) -> dict:
    from app.chat import chat_job

    key = f"chiron_ai:{chat_job._meta_key(job_id)}"
    return json.loads(fake.store[key])


async def _fake_done_stream(*_a, **_k):
    yield {"type": "done", "cancelled": False}


class GateTests(unittest.TestCase):
    def setUp(self):
        from app.chat import chat_job

        self.fake = _FakeRedis()
        chat_job.cache._client = self.fake  # type: ignore[attr-defined]
        chat_job._llm_inflight_sem = None
        chat_job._llm_inflight_waiting = 0

    def tearDown(self):
        from app.chat import chat_job

        chat_job._llm_inflight_sem = None
        chat_job._llm_inflight_waiting = 0

    def _patches(self, stream):
        from app.chat import chat_job

        return [
            # chat_job 已下沉到 services（不再依赖 controllers），mock 目标随之改变
            mock.patch.object(
                chat_job,
                "get_accessible_agent",
                new=mock.AsyncMock(return_value=object()),
            ),
            mock.patch.object(chat_job, "iter_chat_stream_events", stream),
            mock.patch(
                "app.services.user_agent_recent.touch_recent_agent",
                new=mock.AsyncMock(),
            ),
        ]

    def test_gate_free_no_queued_event(self):
        from app.chat import chat_job

        async def _scenario():
            patches = self._patches(_fake_done_stream)
            for p in patches:
                p.start()
            try:
                await chat_job._run_chat_job(
                    job_id="jf",
                    user_id=1,
                    agent_id=2,
                    session_id="s1",
                    message="hi",
                    use_knowledge_retrieval=False,
                )
            finally:
                for p in patches:
                    p.stop()

        asyncio.run(_scenario())
        types = [e["type"] for e in _events(self.fake, "jf")]
        self.assertNotIn("queued", types)
        self.assertEqual(_meta(self.fake, "jf")["status"], "completed")

    def test_gate_full_emits_queued_then_completes(self):
        from app.chat import chat_job

        async def _scenario():
            sem = asyncio.Semaphore(1)
            await sem.acquire()  # 占满唯一槽位
            chat_job._llm_inflight_sem = sem

            async def _release_soon():
                await asyncio.sleep(0.05)
                sem.release()

            patches = self._patches(_fake_done_stream)
            for p in patches:
                p.start()
            releaser = asyncio.create_task(_release_soon())
            try:
                await chat_job._run_chat_job(
                    job_id="jq",
                    user_id=1,
                    agent_id=2,
                    session_id="s1",
                    message="hi",
                    use_knowledge_retrieval=False,
                )
            finally:
                await releaser
                for p in patches:
                    p.stop()

        asyncio.run(_scenario())
        events = _events(self.fake, "jq")
        self.assertEqual(events[0]["type"], "queued")
        self.assertEqual(events[0]["waiting"], 1)
        self.assertEqual(_meta(self.fake, "jq")["status"], "completed")

    def test_queue_timeout_fails_job_without_generation(self):
        from app.chat import chat_job

        async def _blocked_stream(*_a, **_k):
            raise AssertionError("排队超时后不应进入生成流程")
            yield  # pragma: no cover

        async def _scenario():
            sem = asyncio.Semaphore(1)
            await sem.acquire()  # 永不释放 → 必然排队超时
            chat_job._llm_inflight_sem = sem

            patches = self._patches(_blocked_stream)
            patches.append(mock.patch.object(chat_job, "_queue_timeout", return_value=0.05))
            for p in patches:
                p.start()
            try:
                await chat_job._run_chat_job(
                    job_id="jt",
                    user_id=1,
                    agent_id=2,
                    session_id="s1",
                    message="hi",
                    use_knowledge_retrieval=False,
                )
            finally:
                for p in patches:
                    p.stop()

        asyncio.run(_scenario())
        events = _events(self.fake, "jt")
        self.assertEqual(events[0]["type"], "queued")
        self.assertEqual(events[1]["type"], "error")
        meta = _meta(self.fake, "jt")
        self.assertEqual(meta["status"], "failed")
        self.assertIn("排队等待超时", meta["error"])


class JobTerminalStatusTests(unittest.TestCase):
    """JOB-01：中断与正常完成必须能区分开。

    这条曾经是坑：两者都写 `status="completed"`，任何按状态判断「做完了没」的
    调用方都会把「等你确认」当成「干完了」。U10 基线脚本就这么误判过一次，
    得出「代码没修」的错误结论。前端当时没受影响（它读 approval_pending），
    所以缺陷一直没被发现——唯一的消费者恰好绕开了。
    """

    def setUp(self):
        from app.chat import chat_job

        self.fake = _FakeRedis()
        chat_job.cache._client = self.fake  # type: ignore[attr-defined]
        chat_job._llm_inflight_sem = None
        chat_job._llm_inflight_waiting = 0

    def tearDown(self):
        from app.chat import chat_job

        chat_job._llm_inflight_sem = None
        chat_job._llm_inflight_waiting = 0

    def _run(self, stream):
        from app.chat import chat_job

        async def _scenario():
            patches = [
                mock.patch.object(
                    chat_job, "get_accessible_agent", new=mock.AsyncMock(return_value=object())
                ),
                mock.patch.object(chat_job, "iter_chat_stream_events", stream),
                mock.patch(
                    "app.services.user_agent_recent.touch_recent_agent", new=mock.AsyncMock()
                ),
            ]
            for p in patches:
                p.start()
            try:
                await chat_job._run_chat_job(
                    job_id="jt1",
                    user_id=1,
                    agent_id=2,
                    session_id="s1",
                    message="hi",
                    use_knowledge_retrieval=False,
                )
            finally:
                for p in patches:
                    p.stop()

        asyncio.run(_scenario())
        return _meta(self.fake, "jt1")

    def test_interrupted_job_has_its_own_status(self):
        async def _interrupted_stream(*_a, **_k):
            yield {"type": "content", "content": "做到一半"}
            yield {"type": "done", "cancelled": False, "interrupted": True, "message_id": 7}

        meta = self._run(_interrupted_stream)
        self.assertEqual(meta["status"], "interrupted")
        self.assertTrue(meta["interrupted"])

    def test_normal_completion_is_not_marked_interrupted(self):
        async def _normal_stream(*_a, **_k):
            yield {"type": "content", "content": "答完了"}
            yield {"type": "done", "cancelled": False, "interrupted": False}

        meta = self._run(_normal_stream)
        self.assertEqual(meta["status"], "completed")
        self.assertFalse(meta["interrupted"])


if __name__ == "__main__":
    unittest.main()
