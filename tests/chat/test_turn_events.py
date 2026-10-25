"""对话轮次快照的读写：落库、覆盖、终态收敛、读取与失败降级。

用 sqlite 内存库跑（`ChatTurnEvent.__table__` 单表建表，不碰 PG 专有类型），
因此不依赖外部服务。
"""

from __future__ import annotations

import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.chat import turn_events
from app.core.db_models import ChatTurnEvent


class TurnEventsTests(unittest.TestCase):
    def setUp(self) -> None:
        # StaticPool + 单连接：内存库在多次 session 之间保持可见
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        ChatTurnEvent.__table__.create(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._patch = mock.patch.object(turn_events, "SessionLocal", self.Session)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self.engine.dispose()

    def _rows(self) -> list[ChatTurnEvent]:
        db = self.Session()
        try:
            return db.query(ChatTurnEvent).order_by(ChatTurnEvent.id).all()
        finally:
            db.close()

    def _save_snapshot(self, seq: int, content: str = "abc") -> None:
        turn_events.save_snapshot(
            job_id="j1",
            seq=seq,
            user_id=1,
            agent_id=2,
            session_id="s1",
            content=content,
        )

    def test_save_snapshot_persists(self):
        self._save_snapshot(10, "第一段")
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].kind, turn_events.KIND_SNAPSHOT)
        self.assertEqual(rows[0].seq, 10)
        self.assertEqual(rows[0].payload["content"], "第一段")

    def test_snapshot_overwrites_previous(self):
        """同一 job 多次快照只保留最新一条，避免中途快照堆积。"""
        self._save_snapshot(10, "第一段")
        self._save_snapshot(20, "第一段第二段")
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].seq, 20)
        self.assertEqual(rows[0].payload["content"], "第一段第二段")

    def test_terminal_purges_snapshots(self):
        self._save_snapshot(10, "半截")
        turn_events.save_terminal(
            job_id="j1",
            seq=30,
            user_id=1,
            agent_id=2,
            session_id="s1",
            kind=turn_events.KIND_DONE,
            content="完整回答",
        )
        rows = self._rows()
        self.assertEqual(len(rows), 1, "终态后该 job 只应留 1 行")
        self.assertEqual(rows[0].kind, turn_events.KIND_DONE)
        self.assertEqual(rows[0].payload["content"], "完整回答")

    def test_load_latest_picks_highest_seq(self):
        db = self.Session()
        try:
            db.add_all(
                [
                    ChatTurnEvent(
                        job_id="j1", user_id=1, agent_id=2, session_id="s1",
                        seq=5, kind="snapshot", payload={"content": "旧"},
                    ),
                    ChatTurnEvent(
                        job_id="j1", user_id=1, agent_id=2, session_id="s1",
                        seq=9, kind="snapshot", payload={"content": "新"},
                    ),
                ]
            )
            db.commit()
        finally:
            db.close()
        got = turn_events.load_latest_snapshot("j1")
        self.assertIsNotNone(got)
        self.assertEqual(got["seq"], 9)
        self.assertEqual(got["payload"]["content"], "新")

    def test_load_latest_returns_none_when_absent(self):
        self.assertIsNone(turn_events.load_latest_snapshot("missing"))

    def test_disabled_skips_writes(self):
        with mock.patch.object(turn_events, "is_enabled", return_value=False):
            self._save_snapshot(10)
            turn_events.save_terminal(
                job_id="j1", seq=20, user_id=1, agent_id=2, session_id="s1",
                kind=turn_events.KIND_DONE, content="x",
            )
        self.assertEqual(self._rows(), [])

    def test_write_failure_does_not_raise(self):
        """DB 故障时只记日志，不能把异常抛给生成主流程。"""
        broken = mock.Mock(side_effect=RuntimeError("db down"))
        with mock.patch.object(turn_events, "SessionLocal", broken):
            turn_events.save_snapshot(
                job_id="j1", seq=1, user_id=1, agent_id=2, session_id="s1", content="x"
            )

    def test_purge_for_job(self):
        self._save_snapshot(10)
        turn_events.purge_for_job("j1")
        self.assertEqual(self._rows(), [])

    def test_other_jobs_are_isolated(self):
        self._save_snapshot(10, "j1 的内容")
        turn_events.save_snapshot(
            job_id="j2", seq=11, user_id=1, agent_id=2, session_id="s1", content="j2 的内容"
        )
        turn_events.save_terminal(
            job_id="j2", seq=20, user_id=1, agent_id=2, session_id="s1",
            kind=turn_events.KIND_DONE, content="j2 完整",
        )
        j1 = turn_events.load_latest_snapshot("j1")
        j2 = turn_events.load_latest_snapshot("j2")
        self.assertEqual(j1["payload"]["content"], "j1 的内容")
        self.assertEqual(j2["payload"]["content"], "j2 完整")


if __name__ == "__main__":
    unittest.main()
