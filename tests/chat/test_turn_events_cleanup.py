"""对话快照的清理与监控（U2c）。

用 sqlite 内存库跑：只建 `ChatTurnEvent` 单表，不碰 PG 专有类型，不依赖外部服务。
`table_stats` 的 size 分支只在 PostgreSQL 生效，sqlite 下应为 None——这本身是被测行为之一。
"""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.chat import turn_events
from app.core.db_models import ChatTurnEvent


class TurnEventsCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
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

    def _add(
        self,
        *,
        job_id: str = "j1",
        kind: str = turn_events.KIND_DONE,
        age_days: float = 0,
        user_id: int = 1,
        agent_id: int = 2,
        session_id: str = "s1",
        content: str = "abc",
    ) -> None:
        db = self.Session()
        try:
            db.add(
                ChatTurnEvent(
                    job_id=job_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    seq=10,
                    kind=kind,
                    payload={"content": content},
                    created_at=datetime.utcnow() - timedelta(days=age_days),
                )
            )
            db.commit()
        finally:
            db.close()

    def _rows(self) -> list[ChatTurnEvent]:
        db = self.Session()
        try:
            return db.query(ChatTurnEvent).order_by(ChatTurnEvent.id).all()
        finally:
            db.close()

    # --- purge_expired -----------------------------------------------------

    def test_purge_expired_removes_old_terminal_rows(self):
        """超过保留期的终态行被删除，返回值即删除数。"""
        self._add(job_id="old1", age_days=5)
        self._add(job_id="old2", age_days=10, kind=turn_events.KIND_CANCELLED)
        self._add(job_id="new1", age_days=0)

        deleted = turn_events.purge_expired(retention_days=2)

        self.assertEqual(deleted, 2)
        self.assertEqual([r.job_id for r in self._rows()], ["new1"])

    def test_purge_expired_keeps_rows_within_retention(self):
        """未过期的行不受影响。"""
        self._add(job_id="recent", age_days=1)
        self.assertEqual(turn_events.purge_expired(retention_days=2), 0)
        self.assertEqual(len(self._rows()), 1)

    def test_purge_expired_never_removes_mid_flight_snapshot(self):
        """中途快照即使远超保留期也必须保留——否则断线时无内容可恢复。

        这是本模块唯一一条「反直觉」的约束，单独用例守住。
        """
        self._add(job_id="stale_snap", kind=turn_events.KIND_SNAPSHOT, age_days=30)
        self._add(job_id="stale_done", kind=turn_events.KIND_DONE, age_days=30)

        deleted = turn_events.purge_expired(retention_days=2)

        self.assertEqual(deleted, 1)
        remaining = [r.job_id for r in self._rows()]
        self.assertEqual(remaining, ["stale_snap"])

    def test_purge_expired_zero_days_is_noop(self):
        """retention_days=0 表示不清理。"""
        self._add(job_id="old", age_days=100)
        self.assertEqual(turn_events.purge_expired(retention_days=0), 0)
        self.assertEqual(len(self._rows()), 1)

    def test_purge_expired_disabled_is_noop(self):
        """全局开关关闭时什么都不删。"""
        self._add(job_id="old", age_days=100)
        with mock.patch.object(turn_events, "is_enabled", return_value=False):
            self.assertEqual(turn_events.purge_expired(retention_days=1), 0)
        self.assertEqual(len(self._rows()), 1)

    # --- purge_for_session -------------------------------------------------

    def test_purge_for_session_only_touches_target_session(self):
        """按 (user, agent, session) 维度精确删除，不误伤其他会话与用户。"""
        self._add(job_id="a", user_id=1, agent_id=2, session_id="s1")
        self._add(job_id="b", user_id=1, agent_id=2, session_id="s1")
        self._add(job_id="c", user_id=1, agent_id=2, session_id="s2")
        self._add(job_id="d", user_id=9, agent_id=2, session_id="s1")

        deleted = turn_events.purge_for_session(1, 2, "s1")

        self.assertEqual(deleted, 2)
        self.assertEqual(sorted(r.job_id for r in self._rows()), ["c", "d"])

    # --- table_stats / 循环 ------------------------------------------------

    def test_table_stats_reports_rows_and_none_size_on_sqlite(self):
        """sqlite 无 pg_total_relation_size：rows 正常给出，size 为 None 而不是报错。"""
        self._add(job_id="x")
        self._add(job_id="y")
        stats = turn_events.table_stats()
        self.assertIsNotNone(stats)
        self.assertEqual(stats["rows"], 2)
        self.assertIsNone(stats["size_bytes"])

    def test_cleanup_loop_returns_immediately_when_disabled(self):
        """开关关闭时循环立刻返回，不会卡在首次延迟的 sleep 上。"""
        with mock.patch.object(turn_events, "is_enabled", return_value=False), mock.patch.object(
            turn_events, "_cleanup_first_delay", return_value=3600.0
        ):
            # 若实现有误（先 sleep 再判开关），这里会挂住 1 小时；用 wait_for 兜底
            asyncio.run(asyncio.wait_for(turn_events.run_cleanup_loop(), timeout=1.0))

    def test_cleanup_loop_cancels_cleanly(self):
        """被 cancel 时抛 CancelledError 并由调用方吞掉，不产生额外异常。

        lifespan 关闭时就是 `task.cancel()` 一句话，若协程把 CancelledError 当成
        普通异常捕获并继续跑，进程退出会卡住或刷 "Task exception was never retrieved"。
        """

        async def _run() -> None:
            with mock.patch.object(turn_events, "_cleanup_first_delay", return_value=60.0):
                task = asyncio.create_task(turn_events.run_cleanup_loop())
                await asyncio.sleep(0.05)  # 让它进入 sleep
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertTrue(task.cancelled())

        asyncio.run(_run())

    def test_cleanup_once_logs_and_deletes(self):
        """一轮清理：既删除过期行，也不因统计失败而抛异常。"""
        self._add(job_id="old", age_days=9)
        self._add(job_id="new", age_days=0)
        with mock.patch.object(turn_events, "purge_expired", wraps=turn_events.purge_expired) as spy:
            turn_events._cleanup_once()
        spy.assert_called_once()
        self.assertEqual([r.job_id for r in self._rows()], ["new"])


if __name__ == "__main__":
    unittest.main()
