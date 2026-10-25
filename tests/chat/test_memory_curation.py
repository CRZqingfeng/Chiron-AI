"""U7 跨会话记忆：仓储 upsert 语义 + curation 提炼对账。

分两组，都不依赖外部服务：
- `MemoryStoreSqliteTests`：用内存 sqlite 驱动仓储，验「按 mem_key upsert」这套核心语义
- `CurationUnitTests`：mock 掉 LLM 与 Milvus，验提炼流程的容错与水位线

为什么 upsert 语义值得单独立测：这是整个 U7 最容易做歪的地方。
做成追加就是流水账（"我用MySQL"和"我换PG了"并存），做成 delete+insert 就丢了可追溯性。
正确的形态是「旧值标 superseded、新值 active、active 同 key 只能一条」——
而这靠的是**部分唯一索引**，不是普通唯一约束（否则两条同 key 的行无法共存）。
"""

from __future__ import annotations

import unittest
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.chat.memory_store as store_mod
from app.chat.memory_curation import parse_items
from app.core.db_models import ChatMemoryItem


class MemoryStoreSqliteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        ChatMemoryItem.__table__.create(bind=engine)
        cls._orig = store_mod.SessionLocal
        store_mod.SessionLocal = sessionmaker(
            bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
        )

    @classmethod
    def tearDownClass(cls):
        store_mod.SessionLocal = cls._orig

    def setUp(self):
        db = store_mod.SessionLocal()
        try:
            db.query(ChatMemoryItem).delete()
            db.commit()
        finally:
            db.close()

    # ---- upsert -------------------------------------------------------

    def test_create_then_unchanged(self):
        r1 = store_mod.upsert_memory(1, 2, mem_key="db", text="主力数据库是 MySQL")
        self.assertEqual(r1["action"], "created")
        r2 = store_mod.upsert_memory(1, 2, mem_key="db", text="主力数据库是 MySQL")
        self.assertEqual(r2["action"], "unchanged")
        self.assertEqual(r1["id"], r2["id"])

    def test_update_supersedes_old_value(self):
        """换数据库这件事：新值生效、旧值留档，不能两条都 active。"""
        old = store_mod.upsert_memory(1, 2, mem_key="db", text="主力数据库是 MySQL")
        new = store_mod.upsert_memory(1, 2, mem_key="db", text="主力数据库已换成 PostgreSQL")
        self.assertEqual(new["action"], "updated")
        self.assertEqual(new["superseded_id"], old["id"])

        active = store_mod.list_active(1, 2)
        self.assertEqual(len(active), 1, "同 key 的 active 只能有一条")
        self.assertIn("PostgreSQL", active[0]["text"])

        # 旧值仍然查得到（可追溯），只是不再是 active
        old_row = store_mod.get_item(1, 2, old["id"])
        self.assertEqual(old_row["status"], "superseded")
        self.assertIn("MySQL", old_row["text"])

    def test_same_key_different_agent_isolated(self):
        store_mod.upsert_memory(1, 2, mem_key="db", text="智能体 2 用 MySQL")
        store_mod.upsert_memory(1, 3, mem_key="db", text="智能体 3 用 PostgreSQL")
        self.assertIn("MySQL", store_mod.list_active(1, 2)[0]["text"])
        self.assertIn("PostgreSQL", store_mod.list_active(1, 3)[0]["text"])

    def test_empty_key_or_text_rejected(self):
        self.assertEqual(store_mod.upsert_memory(1, 2, mem_key="", text="x")["action"], "error")
        self.assertEqual(store_mod.upsert_memory(1, 2, mem_key="k", text="  ")["action"], "error")

    def test_confidence_clamped(self):
        store_mod.upsert_memory(1, 2, mem_key="k1", text="t", confidence=5.0)
        store_mod.upsert_memory(1, 2, mem_key="k2", text="t", confidence=-3)
        rows = {r["mem_key"]: r for r in store_mod.list_active(1, 2)}
        self.assertEqual(rows["k1"]["confidence"], 1.0)
        self.assertEqual(rows["k2"]["confidence"], 0.0)

    # ---- 上限淘汰 -----------------------------------------------------

    def test_enforce_limit_evicts_lowest_confidence(self):
        for i in range(5):
            store_mod.upsert_memory(1, 2, mem_key=f"k{i}", text=f"t{i}", confidence=0.1 * i)
        evicted = store_mod.enforce_limit(1, 2, max_active=3)
        self.assertEqual(len(evicted), 2, "5 条上限 3 应淘汰 2 条")
        left = store_mod.list_active(1, 2)
        self.assertEqual(len(left), 3)
        # 留下的是 confidence 最高的三条（k2/k3/k4）
        self.assertEqual(sorted(r["mem_key"] for r in left), ["k2", "k3", "k4"])

    def test_enforce_limit_noop_under_limit(self):
        store_mod.upsert_memory(1, 2, mem_key="k", text="t")
        self.assertEqual(store_mod.enforce_limit(1, 2, max_active=10), [])

    # ---- 待同步与删除 -------------------------------------------------

    def test_unsynced_tracks_new_rows(self):
        r = store_mod.upsert_memory(1, 2, mem_key="k", text="t")
        ids = [x["id"] for x in store_mod.list_unsynced(limit=50)]
        self.assertIn(r["id"], ids)
        store_mod.mark_synced([r["id"]])
        self.assertNotIn(r["id"], [x["id"] for x in store_mod.list_unsynced(limit=50)])

    def test_hard_delete_only_own_rows(self):
        r = store_mod.upsert_memory(1, 2, mem_key="k", text="t")
        self.assertFalse(store_mod.hard_delete(1, 3, r["id"]), "跨智能体不能删")
        self.assertTrue(store_mod.hard_delete(1, 2, r["id"]))
        self.assertIsNone(store_mod.get_item(1, 2, r["id"]))

    def test_purge_agent_returns_ids(self):
        a = store_mod.upsert_memory(1, 2, mem_key="k1", text="t1")
        b = store_mod.upsert_memory(1, 2, mem_key="k2", text="t2")
        store_mod.upsert_memory(1, 3, mem_key="k1", text="other agent")
        ids = store_mod.purge_agent_memories(1, 2)
        self.assertEqual(sorted(ids), sorted([a["id"], b["id"]]))
        self.assertEqual(store_mod.list_active(1, 2), [])
        self.assertEqual(len(store_mod.list_active(1, 3)), 1, "别的智能体不受影响")


class ParseItemsTests(unittest.TestCase):
    """LLM 输出不可靠：解析必须容错，绝不抛。"""

    def test_plain_array(self):
        raw = '[{"key":"db","text":"主力数据库 MySQL","kind":"fact","confidence":0.9}]'
        items = parse_items(raw)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["key"], "db")
        self.assertEqual(items[0]["confidence"], 0.9)

    def test_markdown_fence(self):
        raw = '```json\n[{"key":"db","text":"x"}]\n```'
        self.assertEqual(len(parse_items(raw)), 1)

    def test_prose_around_array(self):
        raw = '我提炼如下：\n[{"key":"db","text":"x"}]\n以上。'
        self.assertEqual(len(parse_items(raw)), 1)

    def test_empty_array(self):
        self.assertEqual(parse_items("[]"), [])
        self.assertEqual(parse_items("没有值得记的，输出 []"), [])
        self.assertEqual(parse_items(""), [])
        self.assertEqual(parse_items("not json"), [])

    def test_key_normalised_and_deduped(self):
        raw = '[{"key":"Pref Style!","text":"a"},{"key":"pref_style","text":"b"}]'
        items = parse_items(raw)
        self.assertEqual(len(items), 1, "规范化后同 key 只保留第一条")
        self.assertEqual(items[0]["key"], "pref_style")

    def test_invalid_kind_falls_back(self):
        items = parse_items('[{"key":"k","text":"t","kind":"随便写的"}]')
        self.assertEqual(items[0]["kind"], "fact")

    def test_skips_items_without_key_or_text(self):
        self.assertEqual(parse_items('[{"key":"","text":"t"},{"key":"k","text":""}]'), [])

    def test_respects_max_items(self):
        raw = "[" + ",".join(f'{{"key":"k{i}","text":"t"}}' for i in range(10)) + "]"
        self.assertLessEqual(len(parse_items(raw)), 6)


class CurationFlowTests(unittest.TestCase):
    """提炼流程：水位线只推进在「真提炼过」之后，LLM 挂了不能吞掉这批轮次。"""

    def setUp(self):
        self.pending = [[object(), object()] for _ in range(6)]  # 6 轮，够阈值
        self.keys = list(range(100, 106))

    def _run(self, *, items, pending=None, keys=None):
        from app.chat.memory_curation import curate_session_memories

        with (
            mock.patch(
                "app.chat.memory_curation.pending_turns_for_curation",
                return_value=(self.pending if pending is None else pending, self.keys if keys is None else keys),
            ),
            mock.patch("app.chat.memory_curation._turns_text_for", return_value="对话文本"),
            mock.patch("app.chat.memory_curation._existing_memory_block", return_value="（暂无）"),
            mock.patch("app.chat.memory_curation.extract_items", return_value=items),
            mock.patch(
                "app.chat.memory_curation.upsert_memory",
                return_value={"action": "created", "id": 1, "superseded_id": None},
            ),
            mock.patch("app.chat.memory_curation.enforce_limit", return_value=[]),
            mock.patch("app.chat.memory_curation.list_unsynced", return_value=[]),
            mock.patch("app.chat.storage.storage.get_session_metadata", return_value={}),
            mock.patch("app.chat.storage.storage.patch_session_metadata") as patch_meta,
        ):
            stats = curate_session_memories(1, 2, "s", {"api_key": "k"})
        return stats, patch_meta

    def test_below_threshold_skips_without_touching_watermark(self):
        stats, patch_meta = self._run(
            items=[], pending=[[object()] for _ in range(2)], keys=[1, 2]
        )
        self.assertIsNotNone(stats["skipped"])
        patch_meta.assert_not_called()

    def test_success_advances_watermark(self):
        items = [{"key": "db", "text": "MySQL", "kind": "fact", "confidence": 0.9}]
        stats, patch_meta = self._run(items=items)
        self.assertEqual(stats["created"], 1)
        patch_meta.assert_called_once()
        payload = patch_meta.call_args.args[3]
        self.assertIn("memory_curated_turn_keys", payload)
        self.assertEqual(sorted(payload["memory_curated_turn_keys"]), sorted(self.keys))

    def test_llm_returns_empty_still_records_watermark(self):
        """模型判定「没有值得记的」也是有效结论，不该每轮重复问一遍。"""
        stats, patch_meta = self._run(items=[])
        self.assertEqual(stats["candidates"], 0)
        patch_meta.assert_called_once()

    def test_watermark_merges_with_existing(self):
        """已有水位线不能被覆盖掉，只能合并——否则老轮次会被重复提炼。"""
        from app.chat.memory_curation import curate_session_memories

        with (
            mock.patch(
                "app.chat.memory_curation.pending_turns_for_curation",
                return_value=(self.pending, self.keys),
            ),
            mock.patch("app.chat.memory_curation._turns_text_for", return_value="t"),
            mock.patch("app.chat.memory_curation._existing_memory_block", return_value="（暂无）"),
            mock.patch("app.chat.memory_curation.extract_items", return_value=[]),
            mock.patch("app.chat.memory_curation.list_unsynced", return_value=[]),
            mock.patch(
                "app.chat.storage.storage.get_session_metadata",
                return_value={"memory_curated_turn_keys": [7, 8]},
            ),
            mock.patch("app.chat.storage.storage.patch_session_metadata") as patch_meta,
        ):
            curate_session_memories(1, 2, "s", {"api_key": "k"})
        merged = patch_meta.call_args.args[3]["memory_curated_turn_keys"]
        self.assertIn(7, merged)
        self.assertIn(8, merged)
        for k in self.keys:
            self.assertIn(k, merged)


if __name__ == "__main__":
    unittest.main()
