"""对话轮次的「内容状态快照」读写。

为什么只存快照而不存事件流，见 `db_models.ChatTurnEvent` 的类注释：
一轮对话是 token 级事件（单轮 300–1500 条），逐条落库会吃掉流式延迟；
而前端 `since_seq` 是本地自增计数器，seq 不能有空洞。所以存「某时刻的累积文本」。

设计约束（容易写错的地方）：
- 所有函数**从不抛异常**。快照是尽力而为的兜底，写失败绝不能影响正在进行的生成，
  因此内部 try/except 只记日志，调用方不需要包 try。
- 每次调用**独立 session 且用完立即关闭**：快照是低频任务，不占长连接；
  若复用请求级 session，后台任务与请求生命周期耦合，容易出现「session 已关仍写入」。
- 终态写入后**删除该 job 的所有 snapshot**，保证每个 job 最终只留 1 行；
  否则表会随「快照数 × 对话数」增长。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, select, text

from app.core.db import SessionLocal
from app.core.db_models import ChatTurnEvent
from app.settings import settings

logger = logging.getLogger(__name__)

KIND_SNAPSHOT = "snapshot"
KIND_DONE = "done"
KIND_ERROR = "error"
KIND_CANCELLED = "cancelled"


def is_enabled() -> bool:
    """是否启用快照落库（`CHAT_TURN_EVENTS_ENABLED`）。"""
    return bool(getattr(settings, "CHAT_TURN_EVENTS_ENABLED", True))


def _session():
    return SessionLocal()


def save_snapshot(
    *,
    job_id: str,
    seq: int,
    user_id: int,
    agent_id: int,
    session_id: str,
    content: str,
    thinking: list[Any] | None = None,
) -> None:
    """写入一条中途快照。同 seq 重复写入会覆盖（先删后插），保证幂等。"""
    if not is_enabled():
        return
    db = None
    try:
        db = _session()
        db.execute(
            delete(ChatTurnEvent).where(
                ChatTurnEvent.job_id == job_id,
                ChatTurnEvent.kind == KIND_SNAPSHOT,
            )
        )
        db.add(
            ChatTurnEvent(
                job_id=job_id,
                user_id=int(user_id),
                agent_id=int(agent_id),
                session_id=session_id,
                seq=int(seq),
                kind=KIND_SNAPSHOT,
                payload={"content": content or "", "thinking": thinking or []},
            )
        )
        db.commit()
    except Exception:
        logger.warning("写入对话快照失败 job_id=%s seq=%s", job_id, seq, exc_info=True)
        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def save_terminal(
    *,
    job_id: str,
    seq: int,
    user_id: int,
    agent_id: int,
    session_id: str,
    kind: str,
    content: str,
    thinking: list[Any] | None = None,
    error: str | None = None,
) -> None:
    """写入终态（done / error / cancelled）并清掉该 job 的所有中途快照。

    保留终态一条的意义：断线重连时若 Redis 已丢，客户端仍能拿到完整结果，
    而不是只拿到「生成到一半」的快照。
    """
    if not is_enabled():
        return
    db = None
    try:
        db = _session()
        db.execute(delete(ChatTurnEvent).where(ChatTurnEvent.job_id == job_id))
        db.add(
            ChatTurnEvent(
                job_id=job_id,
                user_id=int(user_id),
                agent_id=int(agent_id),
                session_id=session_id,
                seq=int(seq),
                kind=kind,
                payload={"content": content or "", "thinking": thinking or [], "error": error},
            )
        )
        db.commit()
    except Exception:
        logger.warning(
            "写入对话终态快照失败 job_id=%s kind=%s", job_id, kind, exc_info=True
        )
        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def load_latest_snapshot(job_id: str) -> dict[str, Any] | None:
    """取该 job seq 最大的一条快照；无记录或读取失败返回 None。

    终态的 seq 必然大于任何中途快照（终态在流末尾写入），所以这里
    不需要额外按 kind 过滤。
    """
    if not is_enabled():
        return None
    db = None
    try:
        db = _session()
        row = db.execute(
            select(ChatTurnEvent)
            .where(ChatTurnEvent.job_id == job_id)
            .order_by(ChatTurnEvent.seq.desc(), ChatTurnEvent.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if row is None:
            return None
        return {
            "seq": int(row.seq),
            "kind": row.kind,
            "payload": row.payload or {},
        }
    except Exception:
        logger.warning("读取对话快照失败 job_id=%s", job_id, exc_info=True)
        return None
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def purge_for_job(job_id: str) -> None:
    """删除该 job 的全部快照记录（会话删除/数据清理时使用）。"""
    db = None
    try:
        db = _session()
        db.execute(delete(ChatTurnEvent).where(ChatTurnEvent.job_id == job_id))
        db.commit()
    except Exception:
        logger.warning("清理对话快照失败 job_id=%s", job_id, exc_info=True)
        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 清理与可观测（U2c）
#
# 为什么需要清理：`save_terminal` 让每个 job 收敛为 1 行，但 job 本身是无限增长的
# （每次提问一个新 job_id），长期运行后表会线性膨胀。
#
# 为什么保留期只需 2 天：快照唯一的读取方是 `load_latest_snapshot`，它只被
# `chat_job._resume_snapshot_line` 在 SSE 重连时调用；而进入重连路径的前提是
# job meta 仍在 Redis 里，meta 的 TTL 最长 `CHAT_JOB_TTL_SECONDS`（24h）。
# 也就是说**超过 24 小时的快照在结构上就不可能被读到**，留更久只是积压死数据。
# ---------------------------------------------------------------------------


def _retention_days() -> int:
    """保留天数；<=0 表示不清理（只监控）。"""
    raw = getattr(settings, "CHAT_TURN_EVENTS_RETENTION_DAYS", 2)
    try:
        return int(raw if raw is not None else 2)
    except (TypeError, ValueError):
        return 2


def _cleanup_interval() -> float:
    raw = getattr(settings, "CHAT_TURN_CLEANUP_INTERVAL_SECONDS", 21600)
    try:
        return max(60.0, float(raw if raw is not None else 21600))
    except (TypeError, ValueError):
        return 21600.0


def _cleanup_first_delay() -> float:
    raw = getattr(settings, "CHAT_TURN_CLEANUP_FIRST_DELAY_SECONDS", 300)
    try:
        return max(0.0, float(raw if raw is not None else 300))
    except (TypeError, ValueError):
        return 300.0


def _warn_rows() -> int:
    raw = getattr(settings, "CHAT_TURN_EVENTS_WARN_ROWS", 100_000)
    try:
        return max(1, int(raw if raw is not None else 100_000))
    except (TypeError, ValueError):
        return 100_000


def purge_expired(retention_days: int | None = None) -> int:
    """删除超过保留期的**终态**快照行，返回删除行数。

    只删终态（`kind != snapshot`）：中途快照属于「可能仍在进行中的任务」，
    即使超期也不能删——否则断线时无内容可恢复。这条约束的成本为零，不该省。

    从不抛异常；失败（含未启用、保留期为 0）返回 0。
    """
    if not is_enabled():
        return 0
    days = _retention_days() if retention_days is None else int(retention_days)
    if days <= 0:
        return 0
    # created_at 存的是 naive UTC（见 db_models 的 default=datetime.utcnow），
    # 所以 cutoff 也必须用 naive UTC，否则比较会整体偏移时区。
    # 这里不用 datetime.utcnow()：它自 3.12 起已弃用，显式取 UTC 再去掉 tzinfo 与之等价。
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    db = None
    try:
        db = _session()
        result = db.execute(
            delete(ChatTurnEvent).where(
                ChatTurnEvent.kind != KIND_SNAPSHOT,
                ChatTurnEvent.created_at < cutoff,
            )
        )
        db.commit()
        return int(result.rowcount or 0)
    except Exception:
        logger.warning("清理过期对话快照失败 retention_days=%s", days, exc_info=True)
        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass
        return 0
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def purge_for_session(user_id: int, agent_id: int, session_id: str) -> int:
    """删除某会话的全部快照记录，返回删除行数。

    为什么不能复用 `purge_for_job`：删会话时只有 (user_id, agent_id, session_id)，
    拿不到该会话下曾经产生过的 job_id 列表。若不按这个维度清，用户主动删除的会话
    会在快照表里留下生成内容全文——定时清理只能兜「过期」，兜不住「主动删除」。
    """
    if not is_enabled():
        return 0
    db = None
    try:
        db = _session()
        result = db.execute(
            delete(ChatTurnEvent).where(
                ChatTurnEvent.user_id == int(user_id),
                ChatTurnEvent.agent_id == int(agent_id),
                ChatTurnEvent.session_id == session_id,
            )
        )
        db.commit()
        return int(result.rowcount or 0)
    except Exception:
        logger.warning(
            "清理会话快照失败 user_id=%s agent_id=%s session_id=%s",
            user_id,
            agent_id,
            session_id,
            exc_info=True,
        )
        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass
        return 0
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def table_stats() -> dict[str, Any] | None:
    """返回 {"rows": int, "size_bytes": int | None}；读取失败返回 None。

    size_bytes 用 `pg_total_relation_size`（含索引与 TOAST），仅 PostgreSQL 可用；
    sqlite（单测环境）下为 None。rows 始终尽力给出。
    """
    db = None
    try:
        db = _session()
        bind = db.get_bind()
        dialect = getattr(getattr(bind, "dialect", None), "name", "")
        rows = int(
            db.execute(text("SELECT count(*) FROM mg_chat_turn_events")).scalar() or 0
        )
        size_bytes: int | None = None
        if dialect == "postgresql":
            size_bytes = int(
                db.execute(
                    text("SELECT pg_total_relation_size('mg_chat_turn_events')")
                ).scalar()
                or 0
            )
        return {"rows": rows, "size_bytes": size_bytes}
    except Exception:
        logger.debug("读取对话快照表统计失败（忽略）", exc_info=True)
        return None
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _cleanup_once() -> None:
    """执行一轮快照清理并输出统计。同步函数，由循环用 to_thread 调用。

    checkpoint 的清理不在这里——它是 async（池是 AsyncConnectionPool），
    由 `run_cleanup_loop` 单独 await。
    """
    deleted = purge_expired()
    stats = table_stats()
    if stats is None:
        return
    rows = stats["rows"]
    size_bytes = stats["size_bytes"]
    size_text = f"{size_bytes / 1024 / 1024:.1f}MB" if size_bytes is not None else "n/a"
    # 超过阈值降级为 warning：表异常增长时应当能在日志里被看见，否则只能靠人工查库
    log = logger.warning if rows > _warn_rows() else logger.info
    log(
        "对话快照表清理完成：删除 %s 行，现存 %s 行，体积 %s，保留 %s 天",
        deleted,
        rows,
        size_text,
        _retention_days(),
    )


async def _cleanup_checkpoints() -> None:
    """清理过期 checkpoint 线程。失败只记日志。"""
    try:
        from app.chat.checkpoint import purge_expired_threads

        purged = await purge_expired_threads()
        if purged:
            logger.info("过期 checkpoint 线程清理完成：删除 %s 个", purged)
    except Exception:  # noqa: BLE001
        logger.warning("checkpoint 清理异常（忽略，下一轮继续）", exc_info=True)


async def run_cleanup_loop() -> None:
    """周期清理 + 监控，由 lifespan 启动，随应用关闭被 cancel。

    永不抛异常：任何一轮失败只记日志，下一轮继续——清理是维护性任务，
    绝不能因为它自己出错而影响正在进行的对话。
    """
    if not is_enabled():
        logger.info("对话快照未启用（CHAT_TURN_EVENTS_ENABLED=false），跳过清理任务")
        return
    first_delay = _cleanup_first_delay()
    if first_delay > 0:
        # 避开启动期的 Milvus / 对象存储预热，别在服务还没就绪时发 SQL
        await asyncio.sleep(first_delay)
    interval = _cleanup_interval()
    while True:
        try:
            await asyncio.to_thread(_cleanup_once)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("对话快照清理循环异常（忽略，下一轮继续）", exc_info=True)
        # checkpoint 清理是 async（AsyncConnectionPool 不支持同步 with），单独 await
        await _cleanup_checkpoints()
        await asyncio.sleep(interval)
