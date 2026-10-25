"""SSE 追更：正常重放、按 seq 去重、Redis 丢失后的快照回落。

这里不碰真实 Redis/PG，只验证 `iter_job_sse_events` 的分支决策。
"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

from app.chat import chat_job


def _ev(seq: int, content: str) -> str:
    return json.dumps({"seq": seq, "data": {"type": "content", "content": content}}, ensure_ascii=False)


def _collect(job_id: str, since_seq: int) -> list[str]:
    async def _run():
        out = []
        async for line in chat_job.iter_job_sse_events(job_id, since_seq=since_seq):
            out.append(line)
        return out

    return asyncio.run(_run())


class SseResumeTests(unittest.TestCase):
    def test_normal_path_yields_all_events(self):
        with mock.patch.object(
            chat_job.cache, "lrange_str", side_effect=[[_ev(0, "你"), _ev(1, "好")], []]
        ), mock.patch.object(chat_job.cache, "get_json", return_value={"status": "completed"}):
            lines = _collect("j1", 0)
        self.assertEqual(len(lines), 3, "两条事件 + [DONE]")
        self.assertIn("你", lines[0])
        self.assertIn("好", lines[1])
        self.assertTrue(lines[-1].strip().endswith("[DONE]"))

    def test_skips_already_consumed_by_seq(self):
        """Redis 重启后下标与 seq 错位时，已消费过的事件不能重发。"""
        with mock.patch.object(
            chat_job.cache,
            "lrange_str",
            side_effect=[[_ev(0, "旧"), _ev(1, "新")], []],
        ), mock.patch.object(chat_job.cache, "get_json", return_value={"status": "completed"}):
            lines = _collect("j1", 1)
        self.assertEqual(len(lines), 2, "只应得到 seq=1 那条 + [DONE]")
        self.assertNotIn("旧", lines[0])
        self.assertIn("新", lines[0])

    def test_falls_back_when_redis_meta_missing(self):
        """Redis 整个丢了（连 meta 都没有）→ 用 PG 快照收尾。"""
        snap = {"seq": 42, "kind": "done", "payload": {"content": "已经生成的全文"}}
        with mock.patch.object(chat_job.cache, "lrange_str", return_value=[]), mock.patch.object(
            chat_job.cache, "get_json", return_value=None
        ), mock.patch.object(chat_job, "load_latest_snapshot", return_value=snap):
            lines = _collect("j1", 10)
        self.assertEqual(len(lines), 2)
        payload = json.loads(lines[0].removeprefix("data: ").strip())
        self.assertEqual(payload["type"], "resume_snapshot")
        self.assertEqual(payload["content"], "已经生成的全文")
        self.assertTrue(lines[-1].strip().endswith("[DONE]"))

    def test_falls_back_when_list_truncated(self):
        """列表存在但条目数少于客户端已消费位置 → 判定为被重建/过期。"""
        snap = {"seq": 80, "kind": "snapshot", "payload": {"content": "断线前的内容"}}
        with mock.patch.object(chat_job.cache, "lrange_str", return_value=[]), mock.patch.object(
            chat_job.cache, "get_json", return_value={"status": "running", "boot_id": "x"}
        ), mock.patch.object(chat_job.cache, "llen", return_value=3), mock.patch.object(
            chat_job, "_is_job_alive", return_value=True
        ), mock.patch.object(chat_job, "load_latest_snapshot", return_value=snap):
            lines = _collect("j1", 50)
        self.assertEqual(len(lines), 2)
        payload = json.loads(lines[0].removeprefix("data: ").strip())
        self.assertEqual(payload["type"], "resume_snapshot")
        self.assertEqual(payload["content"], "断线前的内容")

    def test_no_snapshot_means_empty_stream(self):
        """Redis 丢了、PG 也没快照 → 只给 [DONE]，不能挂住。"""
        with mock.patch.object(chat_job.cache, "lrange_str", return_value=[]), mock.patch.object(
            chat_job.cache, "get_json", return_value=None
        ), mock.patch.object(chat_job, "load_latest_snapshot", return_value=None):
            lines = _collect("j1", 5)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].strip().endswith("[DONE]"))


if __name__ == "__main__":
    unittest.main()
