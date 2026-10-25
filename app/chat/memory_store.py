"""跨会话记忆条目的仓储层（`mg_chat_memories`）。

只做数据访问，不碰 LLM、不碰 Milvus —— 提炼与双写编排在 `memory_curation.py`。
分层理由：仓储要能被单测用 sqlite 直接驱动，混进 LLM/向量调用就没法离线测了。

核心语义是 **按 `mem_key` upsert**，不是追加：
- 同 key 且正文完全相同 → `unchanged`（不写库、不重排 Milvus）
- 同 key 但正文变了 → 旧行标 `superseded`、插新行（保留历史可追溯）
- 没有同 key → 插新行

易错点：更新时必须**先 flush 把旧行改成 superseded**，再 insert 新行。
两者在同一事务里，PG 的部分唯一索引 `WHERE status='active'` 是逐语句检查的，
顺序反了会撞唯一约束。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.core.db import SessionLocal
from app.core.db_models import ChatMemoryItem

logger = logging.getLogger(__name__)

# 单用户单智能体最多保留多少条 active 记忆。超了淘汰 confidence 最低的。
# 定这个上限的理由：记忆是为了减少重复沟通，不是为了存档案；条数一多，注入的
# top-3 就会被无关条目挤占，反而不如不注入。
DEFAULT_MAX_ACTIVE = 200


def upsert_memory(
    user_id: int,
    agent_id: int,
    *,
    mem_key: str,
    text: str,
    kind: str = "fact",
    confidence: float = 0.8,
    source_session: str | None = None,
    source_turn_key: int | None = None,
) -> dict[str, Any]:
    """按 mem_key 写入/更新一条记忆。

    :return: ``{"action": "created"|"updated"|"unchanged", "id": int, "superseded_id": int|None}``
    """
    key = (mem_key or "").strip()
    body = (text or "").strip()
    if not key or not body:
        return {"action": "error", "id": 0, "reason": "mem_key/text 不能为空"}

    db = SessionLocal()
    try:
        row = (
            db.query(ChatMemoryItem)
            .filter(
                ChatMemoryItem.user_id == user_id,
                ChatMemoryItem.agent_id == agent_id,
                ChatMemoryItem.mem_key == key,
                ChatMemoryItem.status == "active",
            )
            .first()
        )
        if row is not None and (row.text or "").strip() == body:
            return {"action": "unchanged", "id": int(row.id), "superseded_id": None}

        superseded_id = None
        if row is not None:
            superseded_id = int(row.id)
            row.status = "superseded"
            row.embedding_synced = True  # 旧条目不需要再同步（Milvus 里那条按 id 删）
            row.updated_at = datetime.utcnow()
            # 必须先落库，让部分唯一索引看到"active 已让出"再插新行
            db.flush()

        item = ChatMemoryItem(
            user_id=user_id,
            agent_id=agent_id,
            mem_key=key,
            text=body,
            kind=(kind or "fact").strip()[:24],
            confidence=max(0.0, min(1.0, float(confidence))),
            source_session=(source_session or None),
            source_turn_key=int(source_turn_key) if source_turn_key else None,
            status="active",
            embedding_synced=False,
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return {
            "action": "updated" if superseded_id else "created",
            "id": int(item.id),
            "superseded_id": superseded_id,
        }
    except Exception:
        db.rollback()
        logger.exception("upsert_memory failed: user=%s agent=%s key=%s", user_id, agent_id, key)
        return {"action": "error", "id": 0, "reason": "写入失败"}
    finally:
        db.close()


def list_active(user_id: int, agent_id: int, *, limit: int = 50) -> list[dict[str, Any]]:
    """列出生效中的记忆（按更新时间倒序）。"""
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMemoryItem)
            .filter(
                ChatMemoryItem.user_id == user_id,
                ChatMemoryItem.agent_id == agent_id,
                ChatMemoryItem.status == "active",
            )
            .order_by(ChatMemoryItem.updated_at.desc())
            .limit(max(1, int(limit)))
            .all()
        )
        return [_to_dict(r) for r in rows]
    finally:
        db.close()


def list_by_ids(user_id: int, agent_id: int, ids: list[int]) -> dict[int, dict[str, Any]]:
    """按 id 批量取（Milvus 命中后回 PG 补 kind / mem_key 用，避免 N 次单查）。"""
    clean = [int(i) for i in (ids or []) if i]
    if not clean:
        return {}
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMemoryItem)
            .filter(
                ChatMemoryItem.user_id == user_id,
                ChatMemoryItem.agent_id == agent_id,
                ChatMemoryItem.id.in_(clean),
            )
            .all()
        )
        return {int(r.id): _to_dict(r) for r in rows}
    finally:
        db.close()


def get_item(user_id: int, agent_id: int, mem_id: int) -> dict[str, Any] | None:
    db = SessionLocal()
    try:
        row = (
            db.query(ChatMemoryItem)
            .filter(
                ChatMemoryItem.id == int(mem_id),
                ChatMemoryItem.user_id == user_id,
                ChatMemoryItem.agent_id == agent_id,
            )
            .first()
        )
        return _to_dict(row) if row else None
    finally:
        db.close()


def hard_delete(user_id: int, agent_id: int, mem_id: int) -> bool:
    """物理删除一条记忆。

    **调用方必须先删 Milvus 里对应的向量**——行删了就再也拿不到 id 拼 chunk_id 了。
    所以这里不做级联，只负责 PG 侧。
    """
    db = SessionLocal()
    try:
        n = (
            db.query(ChatMemoryItem)
            .filter(
                ChatMemoryItem.id == int(mem_id),
                ChatMemoryItem.user_id == user_id,
                ChatMemoryItem.agent_id == agent_id,
            )
            .delete()
        )
        db.commit()
        return bool(n)
    except Exception:
        db.rollback()
        logger.exception("hard_delete failed: id=%s", mem_id)
        return False
    finally:
        db.close()


def mark_synced(ids: list[int]) -> None:
    if not ids:
        return
    db = SessionLocal()
    try:
        db.query(ChatMemoryItem).filter(ChatMemoryItem.id.in_([int(i) for i in ids])).update(
            {"embedding_synced": True}, synchronize_session=False
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("mark_synced failed: ids=%s", ids)
    finally:
        db.close()


def list_unsynced(*, limit: int = 100) -> list[dict[str, Any]]:
    """待同步到 Milvus 的条目（含已 superseded 的，用来清理旧向量）。"""
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMemoryItem)
            .filter(ChatMemoryItem.embedding_synced.is_(False))
            .order_by(ChatMemoryItem.id)
            .limit(max(1, int(limit)))
            .all()
        )
        return [_to_dict(r) for r in rows]
    finally:
        db.close()


def count_active(user_id: int, agent_id: int) -> int:
    db = SessionLocal()
    try:
        return int(
            db.query(ChatMemoryItem)
            .filter(
                ChatMemoryItem.user_id == user_id,
                ChatMemoryItem.agent_id == agent_id,
                ChatMemoryItem.status == "active",
            )
            .count()
        )
    finally:
        db.close()


def enforce_limit(user_id: int, agent_id: int, *, max_active: int = DEFAULT_MAX_ACTIVE) -> list[int]:
    """active 条数超上限时，把 confidence 最低的标 superseded。

    :return: 被降级的 id 列表（调用方需要把它们从 Milvus 删掉）
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMemoryItem)
            .filter(
                ChatMemoryItem.user_id == user_id,
                ChatMemoryItem.agent_id == agent_id,
                ChatMemoryItem.status == "active",
            )
            .order_by(ChatMemoryItem.confidence.asc(), ChatMemoryItem.updated_at.asc())
            .all()
        )
        overflow = len(rows) - max(1, int(max_active))
        if overflow <= 0:
            return []
        victims = rows[:overflow]
        ids = [int(r.id) for r in victims]
        for r in victims:
            r.status = "superseded"
            r.updated_at = datetime.utcnow()
        db.commit()
        logger.info("记忆条数超上限，降级 %s 条（user=%s agent=%s）", len(ids), user_id, agent_id)
        return ids
    except Exception:
        db.rollback()
        logger.exception("enforce_limit failed")
        return []
    finally:
        db.close()


def purge_agent_memories(user_id: int, agent_id: int) -> list[int]:
    """删除某智能体的全部记忆行（删智能体时调用），返回被删的 id 供清 Milvus。"""
    db = SessionLocal()
    try:
        ids = [
            int(r.id)
            for r in db.query(ChatMemoryItem.id)
            .filter(
                ChatMemoryItem.user_id == user_id,
                ChatMemoryItem.agent_id == agent_id,
            )
            .all()
        ]
        db.query(ChatMemoryItem).filter(
            ChatMemoryItem.user_id == user_id,
            ChatMemoryItem.agent_id == agent_id,
        ).delete(synchronize_session=False)
        db.commit()
        return ids
    except Exception:
        db.rollback()
        logger.exception("purge_agent_memories failed")
        return []
    finally:
        db.close()


def _to_dict(row: ChatMemoryItem) -> dict[str, Any]:
    return {
        "id": int(row.id),
        "user_id": int(row.user_id),
        "agent_id": int(row.agent_id),
        "mem_key": row.mem_key,
        "text": row.text,
        "kind": row.kind,
        "confidence": float(row.confidence or 0.0),
        "source_session": row.source_session,
        "source_turn_key": int(row.source_turn_key) if row.source_turn_key else None,
        "status": row.status,
        "embedding_synced": bool(row.embedding_synced),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
