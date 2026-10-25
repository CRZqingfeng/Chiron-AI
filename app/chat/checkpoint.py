"""LangGraph checkpoint 的持久化入口（审批中断的状态存储）。

为什么需要它：审批中断的本质是「把图停在中途，等人工决定后再从断点继续」。
这件事必须有地方存住中断时的完整状态（messages、tool_calls、待决策项），
否则用户离开十分钟回来点确认，进程一重启就什么都找不到了。

为什么默认 Postgres 而不是内存：`InMemorySaver` 生命周期跟着进程走。
单实例跑着没事，但一次重启就丢，而这恰恰是「审批」最不能接受的失败模式。

**Windows 上的额外约束**：psycopg 的 async 连接池在 `ProactorEventLoop` 下
直接拒绝工作（会刷 "Psycopg cannot use the 'ProactorEventLoop'" 直到超时）。
所以 `run.py` 里把事件循环策略切成了 `WindowsSelectorEventLoopPolicy`。
若某天有人绕过 `run.py` 用 `uvicorn app.main:app`（但没有切事件循环）启动，
这里会自动降级到内存并在日志里说明原因——不静默失败，也不阻断对话。

表结构由 `setup()` 自动建立（`checkpoints` / `checkpoint_blobs` /
`checkpoint_writes` / `checkpoint_migrations`），与现有 `mg_*` 表无交集。
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid

from app.log import logger
from app.settings import settings

_pool = None
_saver = None
_init_error: str | None = None

# 2026-10-24 体检实录：包装齐了但后端仍是内存，原因被日志里的 %s 吃掉，
# 排查时误判成「缺包」。这类失败不能靠捕获异常识别——psycopg 会把真实原因
# 包成 `PoolTimeout: pool initialization incomplete after 15 sec`，
# 关键字里根本看不到 ProactorEventLoop。所以改为**主动探测事件循环类型**。
_PROACTOR_HINT = (
    "原因：psycopg 的 async 连接池不支持 Windows 的 ProactorEventLoop。"
    "请用 `python run.py` 启动（它会切到 SelectorEventLoop）；"
    "若必须手工起服务，先执行 asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())。"
    "注意：uvicorn --reload 的子进程不继承该策略。"
)


def _proactor_loop_active() -> bool:
    """当前运行的事件循环是不是 Proactor（Windows 默认，psycopg async 不兼容）。

    非 Windows 没有 ProactorEventLoop，直接返回 False。
    """
    proactor = getattr(asyncio, "ProactorEventLoop", None)
    if proactor is None:
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    return isinstance(loop, proactor)


def _enabled() -> bool:
    return bool(getattr(settings, "CHAT_APPROVAL_ENABLED", True))


def _backend() -> str:
    return str(getattr(settings, "CHAT_CHECKPOINT_BACKEND", "postgres") or "postgres").strip().lower()


def _to_conninfo(raw: str) -> str:
    """把 SQLAlchemy 风格的 URL 转成 psycopg 原生 conninfo。

    `postgresql+psycopg://user:pwd@host:port/db` → `postgresql://user:pwd@host:port/db`
    """
    return re.sub(r"^postgresql\+[a-z0-9_]+://", "postgresql://", str(raw or ""))


def _conninfo() -> str:
    return _to_conninfo(str(settings.chat_database_url))


def _pool_size() -> tuple[int, int]:
    """连接池大小。checkpoint 只在每轮对话开始时读写几次，不需要大池。"""
    try:
        size = max(1, int(getattr(settings, "CHAT_CHECKPOINT_POOL_SIZE", 2)))
    except (TypeError, ValueError):
        size = 2
    return (1, size)


async def _build_postgres():
    """建连接池 + saver + 表。失败抛异常，由调用方降级。"""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    global _pool
    min_size, max_size = _pool_size()
    _pool = AsyncConnectionPool(
        _conninfo(),
        min_size=min_size,
        max_size=max_size,
        open=False,
        # autocommit 是 langgraph 的要求（它自己管事务边界）；
        # dict_row 让返回行可按列名取，saver 内部依赖这一点
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await _pool.open(wait=True, timeout=15)
    saver = AsyncPostgresSaver(_pool)
    await saver.setup()  # CREATE TABLE IF NOT EXISTS，幂等
    return saver


async def get_checkpointer():
    """返回 checkpointer；未启用或创建失败时返回内存实现（绝不抛异常）。

    调用方拿到 None 表示「本轮不支持审批中断」，需要按老路径走。
    """
    global _saver, _init_error
    if not _enabled():
        return None
    if _saver is not None:
        return _saver

    backend = _backend()
    if backend == "none":
        return None

    if backend == "postgres" and _init_error is None:
        try:
            # 快速失败：ProactorEventLoop 下 psycopg 必然连不上，
            # 别白等 15 秒连接池超时，也别让用户从 PoolTimeout 里猜原因
            if _proactor_loop_active():
                raise RuntimeError(_PROACTOR_HINT)
            _saver = await _build_postgres()
            logger.info("审批中断：checkpoint 已使用 PostgreSQL 持久化")
            return _saver
        except Exception as e:  # noqa: BLE001 - 任何失败都降级，不能阻断对话
            _init_error = f"{type(e).__name__}: {e}"
            # loguru 只认 {}；写成 %s 会把参数连同真正的原因一起丢掉（改前实测过）
            logger.warning(
                "checkpoint 初始化失败，降级为内存存储（重启后未完成的审批会失效）：{}",
                _init_error,
            )
            await close_checkpointer()

    # 内存兜底：进程内有效，够跑通机制验证与本地开发
    from langgraph.checkpoint.memory import InMemorySaver

    _saver = InMemorySaver()
    return _saver


async def close_checkpointer() -> None:
    """释放连接池。lifespan 退出时调用；重复调用安全。"""
    global _pool, _saver
    _saver = None
    if _pool is not None:
        try:
            await _pool.close()
        except Exception:  # noqa: BLE001
            logger.debug("关闭 checkpoint 连接池失败（忽略）", exc_info=True)
        _pool = None


async def reset_for_tests() -> None:
    """测试用：清掉单例，便于在不同 event loop 里重建。"""
    global _init_error
    await close_checkpointer()
    _init_error = None


__all__ = ["get_checkpointer", "close_checkpointer", "reset_for_tests"]


# ---------------------------------------------------------------------------
# 过期线程清理
#
# `checkpoints` 表**没有时间列**（它由 langgraph 管理，不该改表结构），
# 好在 checkpoint_id 是 uuid6——UUID v6 前 60 位就是时间戳，天然有序。
# 反解出来即可按时间清理，无需额外建表记账。
#
# 保守原则：解析失败（不是 v6、格式异常）的线程一律不删。
# 宁可留垃圾，也不能误删一个正等着用户确认的审批。
# ---------------------------------------------------------------------------

# 1582-10-15 到 1970-01-01 的 100 纳秒数，UUID 时间戳的纪元偏移
_UUID_EPOCH_OFFSET = 12219292800


def _uuid6_unix_time(checkpoint_id: str) -> float | None:
    """从 uuid6 反解 Unix 时间戳（秒）；不是 v6 或格式异常时返回 None。"""
    try:
        uid = uuid.UUID(str(checkpoint_id))
    except (ValueError, AttributeError, TypeError):
        return None
    if uid.version != 6:
        return None
    raw = uid.bytes
    time_high = int.from_bytes(raw[0:4], "big")
    time_mid = int.from_bytes(raw[4:6], "big")
    time_low = int.from_bytes(raw[6:8], "big") & 0x0FFF
    hundred_ns = (time_high << 28) | (time_mid << 12) | time_low
    return hundred_ns / 1e7 - _UUID_EPOCH_OFFSET


def _retention_days() -> int:
    raw = getattr(settings, "CHAT_CHECKPOINT_RETENTION_DAYS", 1)
    try:
        return int(raw if raw is not None else 1)
    except (TypeError, ValueError):
        return 1


async def purge_expired_threads(retention_days: int | None = None) -> int:
    """删除超过保留期的 checkpoint 线程，返回删除的线程数。

    ⚠️ 必须是 async：池是 `AsyncConnectionPool`，用同步 `with pool.connection()`
    会直接抛异常（然后被兜底吞掉、静默返回 0，看起来像"没有过期数据"）。
    这是实测踩过的坑。

    从不抛异常；未启用、保留期为 0、或池未建立时返回 0。
    """
    if _pool is None:
        return 0
    days = _retention_days() if retention_days is None else int(retention_days)
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400

    try:
        async with _pool.connection() as conn:
            cur = await conn.execute(
                "SELECT thread_id, max(checkpoint_id) AS latest "
                "FROM checkpoints GROUP BY thread_id"
            )
            rows = await cur.fetchall()
            doomed: list[str] = []
            for row in rows:
                # dict_row → 可按列名取；兼容元组风格的兜底
                try:
                    thread_id = row["thread_id"]
                    latest = row["latest"]
                except (TypeError, KeyError, IndexError):
                    thread_id, latest = row[0], row[1]
                created = _uuid6_unix_time(latest)
                if created is not None and created < cutoff:
                    doomed.append(thread_id)
            if not doomed:
                return 0
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                await conn.execute(f"DELETE FROM {table} WHERE thread_id = ANY(%s)", (doomed,))
            return len(doomed)
    except Exception:  # noqa: BLE001 - 清理失败不该影响任何业务
        logger.warning("清理过期 checkpoint 失败（忽略）", exc_info=True)
        return 0
