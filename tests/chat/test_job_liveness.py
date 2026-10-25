"""Job 存活判定：心跳与 boot_id 决定一个 running 状态是否可信。

背景：job 是进程内 asyncio task（见 create_chat_job），进程被杀或重启后 task 消失，
但 Redis 里的 meta 仍带着 status=running 活 24h。若只按状态判断，该会话会被永久
判为「有任务在跑」——新建 409、切分支 409、SSE 追更永不退出。
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from unittest import mock

from app.chat import chat_job


class _FakeRedis:
    """最小 Redis 替身：只实现 chat_job 用到的几个命令。"""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return False
        self.store[key] = value
        return True

    def get(self, key):
        return self.store.get(key)

    def exists(self, key):
        return 1 if key in self.store else 0

    def setex(self, key, ttl, value):
        self.store[key] = value
        return True

    def delete(self, key):
        self.store.pop(key, None)
        return 1

    def eval(self, *args, **kwargs):
        return 0

    def expire(self, *args, **kwargs):
        return True

    def rpush(self, *args, **kwargs):
        return 1

    def lrange(self, *args, **kwargs):
        return []

    def llen(self, *args, **kwargs):
        return 0


def _meta(**over):
    """构造一份「本进程刚创建」的 job meta，允许覆盖任意字段。"""
    base = {
        "job_id": "j1",
        "user_id": 1,
        "agent_id": 2,
        "session_id": "s1",
        "status": "running",
        "boot_id": chat_job._BOOT_ID,
        "created_at": time.time(),
    }
    base.update(over)
    return base


class JobLivenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = _FakeRedis()
        chat_job.cache._client = self.fake  # type: ignore[attr-defined]

    def _put_hb(self, job_id: str = "j1") -> None:
        key = chat_job.cache._key(chat_job._hb_key(job_id))
        self.fake.store[key] = '{"boot_id": "x"}'

    def _del_hb(self, job_id: str = "j1") -> None:
        self.fake.store.pop(chat_job.cache._key(chat_job._hb_key(job_id)), None)

    def test_alive_when_hb_present_and_same_boot(self):
        self._put_hb()
        self.assertTrue(chat_job._is_job_alive("j1", _meta()))

    def test_alive_when_hb_missing_but_same_boot(self):
        """本进程创建的 job，即使心跳因 Redis 抖动没写成功，也不能判死。"""
        self._del_hb()
        self.assertTrue(chat_job._is_job_alive("j1", _meta()))

    def test_dead_when_hb_missing_and_boot_differs(self):
        """心跳没了 + 不是本进程创建的 → 上个进程遗留的 running。"""
        self._del_hb()
        self.assertFalse(chat_job._is_job_alive("j1", _meta(boot_id="old-boot")))

    def test_dead_when_hb_present_but_boot_differs_single_instance(self):
        """旧进程的心跳还没过期，但单实例部署下能认出那不是自己的。"""
        self._put_hb()
        with mock.patch.object(chat_job, "_single_instance", return_value=True):
            self.assertFalse(chat_job._is_job_alive("j1", _meta(boot_id="old-boot")))

    def test_alive_when_boot_differs_multi_instance(self):
        """多实例下不能误杀其他实例正在跑的任务。"""
        self._put_hb()
        with mock.patch.object(chat_job, "_single_instance", return_value=False):
            self.assertTrue(chat_job._is_job_alive("j1", _meta(boot_id="other-instance")))

    def test_redis_unavailable_returns_alive(self):
        """Redis 无法判断时必须保守放行，不能误杀。"""
        with mock.patch.object(chat_job.cache, "exists", return_value=None):
            self.assertTrue(chat_job._is_job_alive("j1", _meta(boot_id="old-boot")))

    def test_terminal_status_returns_false(self):
        for status in ("completed", "failed", "cancelled"):
            self.assertFalse(chat_job._is_job_alive("j1", _meta(status=status)))

    def test_missing_meta_returns_false(self):
        with mock.patch.object(chat_job, "get_job_meta", return_value=None):
            self.assertFalse(chat_job._is_job_alive("j1"))

    def test_reap_marks_failed_and_writes_error_event(self):
        """孤儿收敛：写一条 error 事件、置 failed、释放会话锁。"""
        meta = _meta(boot_id="old-boot")
        captured: dict = {}

        async def _fake_append(job_id, seq, data):
            captured["job_id"] = job_id
            captured["seq"] = seq
            captured["data"] = data

        with mock.patch.object(chat_job, "_append_event", _fake_append), mock.patch.object(
            chat_job, "_release_active_key"
        ) as release:
            asyncio.run(chat_job._reap_dead_job("j1", meta))

        self.assertEqual(captured["job_id"], "j1")
        self.assertEqual(captured["data"]["type"], "error")
        self.assertEqual(captured["data"]["content"], chat_job._ORPHAN_REASON)
        self.assertEqual(meta["status"], "failed")
        self.assertEqual(meta["error"], chat_job._ORPHAN_REASON)
        release.assert_called_once()

    def test_reap_respects_cancel_flag(self):
        """用户主动停止的任务不写错误事件，状态收敛为 cancelled。"""
        meta = _meta(boot_id="old-boot")
        append = mock.AsyncMock()
        with mock.patch.object(chat_job, "is_job_cancel_requested", return_value=True), mock.patch.object(
            chat_job, "_append_event", append
        ), mock.patch.object(chat_job, "_release_active_key"):
            asyncio.run(chat_job._reap_dead_job("j1", meta))

        append.assert_not_called()
        self.assertEqual(meta["status"], "cancelled")
        self.assertIsNone(meta["error"])

    def test_create_job_reaps_orphan_then_starts_new(self):
        """核心主路径：上个进程遗留的 running 不应再挡住用户新建任务。

        这是 U0 的验收本体——把 active 锁与 meta 预置成「旧 boot 的 running」，
        模拟进程重启后的现场，断言 create_chat_job 会先收敛旧任务再开出新 job。
        """
        ak = chat_job._active_key(1, 2, "s1")
        self.fake.store[chat_job.cache._key(ak)] = json.dumps({"job_id": "old"})
        old_meta = _meta(job_id="old", boot_id="old-boot", created_at=time.time() - 60)
        self.fake.store[chat_job.cache._key(chat_job._meta_key("old"))] = json.dumps(old_meta)

        def _fake_release(user_id, agent_id, session_id, job_id):
            self.fake.store.pop(chat_job.cache._key(chat_job._active_key(user_id, agent_id, session_id)), None)

        with mock.patch.object(chat_job, "_release_active_key", _fake_release), mock.patch.object(
            chat_job.asyncio, "create_task", lambda coro: coro.close() or mock.Mock()
        ):
            new_id, is_dup = asyncio.run(
                chat_job.create_chat_job(
                    user_id=1,
                    agent_id=2,
                    session_id="s1",
                    message="hi",
                    use_knowledge_retrieval=False,
                )
            )

        self.assertFalse(is_dup, "旧任务已死，不应判定为重复")
        self.assertNotEqual(new_id, "old")
        old_after = json.loads(self.fake.store[chat_job.cache._key(chat_job._meta_key("old"))])
        self.assertEqual(old_after["status"], "failed", "旧任务应被收敛为失败")
        self.assertEqual(old_after["error"], chat_job._ORPHAN_REASON)


if __name__ == "__main__":
    unittest.main()
