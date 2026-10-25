"""
智能体流式对话异步 Job：后台执行生成，事件写入 Redis，支持断线后按 seq 重连 SSE。
用于将对话任务放在后台执行，不会被用户的其它请求打断。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from app.chat.agent_service import iter_chat_stream_events
from app.core.cache import cache
from app.chat.preview_session import is_editor_preview_session
from app.chat.turn_events import (
    KIND_CANCELLED,
    KIND_DONE,
    KIND_ERROR,
    is_enabled as turn_events_enabled,
    load_latest_snapshot,
    save_snapshot,
    save_terminal,
)
from app.services.user_agent_access import get_accessible_agent
from app.log import logger
from app.settings import settings

# 本进程实例标识。job 的 meta 里会记下创建它的进程，判定存活时用它区分
# "本进程创建的" 与 "上个进程遗留的 running"（后者执行体早已随进程消失）。
_BOOT_ID = uuid.uuid4().hex

# 孤儿任务的用户可见文案：进程被杀或重启后，执行体已不存在，本轮不可能继续
_ORPHAN_REASON = "连接中断，请重试"

# SSE 追更时校验「执行体是否还活着」的节流间隔（秒）：
# 追更循环本身是 40–300ms 一轮，不能每圈都读心跳键
_LIVENESS_CHECK_INTERVAL = 3.0

_llm_inflight_sem: asyncio.Semaphore | None = None
# 正在等待并发闸门的 Job 数（仅用于排队提示，不参与调度）
_llm_inflight_waiting: int = 0


def _llm_inflight() -> asyncio.Semaphore:
    global _llm_inflight_sem
    if _llm_inflight_sem is None:
        n = max(1, int(getattr(settings, "LLM_MAX_INFLIGHT", 8) or 8))
        _llm_inflight_sem = asyncio.Semaphore(n)
    return _llm_inflight_sem


def _queue_timeout() -> float:
    """并发闸门排队等待上限（秒），超时置任务失败而非无限静默排队。"""
    return max(1.0, float(getattr(settings, "LLM_QUEUE_TIMEOUT_SECONDS", 120) or 120))


def _meta_key(job_id: str) -> str:
    """
    获取 Job 元数据 key
    """
    return f"chat_job:{job_id}:meta"


def _events_key(job_id: str) -> str:
    """
    获取 Job 事件 key
    """
    return f"chat_job:{job_id}:events"


def _active_key(user_id: int, agent_id: int, session_id: str) -> str:
    """
    获取 Job 活动 key
    """
    return f"chat_job_active:{user_id}:{agent_id}:{session_id}"


def _cancel_key(job_id: str) -> str:
    """用户请求停止生成时写入的标记 key。"""
    return f"chat_job:{job_id}:cancel"


def _hb_key(job_id: str) -> str:
    """Job 心跳键：存在即表示该 job 的执行体仍在运行（由 _heartbeat_loop 周期刷新）。"""
    return f"chat_job:{job_id}:hb"


def is_job_cancel_requested(job_id: str) -> bool:
    """是否已请求取消该 Job（同步读缓存，供 iter 内 to_thread 调用）。"""
    raw = cache.get_json(_cancel_key(job_id))
    return bool(raw)


def get_running_session_job(user_id: int, agent_id: int, session_id: str) -> dict | None:
    """返回该会话当前**真实在跑**的 Job 元数据；无则 None（用于切分支前的冲突检查）。

    这里是同步函数（被路由直接调用），只做判定不自愈：
    自愈需要写 Redis，留给下一次 create_chat_job，避免在读路径上产生写副作用。
    """
    existing = cache.get_json(_active_key(user_id, agent_id, session_id))
    if not isinstance(existing, dict) or not existing.get("job_id"):
        return None
    job_id = str(existing["job_id"])
    meta = get_job_meta(job_id)
    if not meta or meta.get("status") != "running":
        return None
    # running 但执行体已消失 → 不计为「进行中」，否则切分支会被永久 409
    return meta if _is_job_alive(job_id, meta) else None


def _release_active_key(user_id: int, agent_id: int, session_id: str, job_id: str) -> None:
    """
    释放会话占用锁：仅当锁仍指向该 job 时才删除，
    避免旧任务退出时误删新任务的占用锁。
    """
    cache.delete_if_job_matches(_active_key(user_id, agent_id, session_id), job_id)

async def request_chat_job_cancel(job_id: str) -> None:
    """
    标记 Job 为「用户请求停止」并即时终结：
    1. 写取消标记，供生成协程协作中断；
    2. 若任务仍 running，立即置为 cancelled 并释放会话占用锁，
       使用户停止后可立刻发起新任务，不必等待旧任务完全退出。
    """
    await asyncio.to_thread(cache.set_json, _cancel_key(job_id), {"v": 1}, _ttl())
    meta = await asyncio.to_thread(cache.get_json, _meta_key(job_id))
    if not isinstance(meta, dict) or meta.get("status") != "running":
        return
    meta["status"] = "cancelled"
    meta["error"] = None
    await asyncio.to_thread(cache.set_json, _meta_key(job_id), meta, _ttl())
    await asyncio.to_thread(
        _release_active_key,
        int(meta.get("user_id", 0)),
        int(meta.get("agent_id", 0)),
        str(meta.get("session_id", "")),
        job_id,
    )

async def cancel_active_session_job(user_id: int, agent_id: int, session_id: str) -> bool:
    """
    按会话取消当前活动任务（前端停止时 job_id 未知的兜底，如创建请求在途被中断）。
    :return: 是否实际取消了任务
    """
    existing = await asyncio.to_thread(cache.get_json, _active_key(user_id, agent_id, session_id))
    if not isinstance(existing, dict) or not existing.get("job_id"):
        return False
    job_id = str(existing["job_id"])
    meta = await asyncio.to_thread(cache.get_json, _meta_key(job_id))
    if not meta or int(meta.get("user_id", -1)) != int(user_id):
        return False
    if meta.get("status") != "running":
        return False
    await request_chat_job_cancel(job_id)
    return True


def _ttl() -> int:
    """
    获取 Job 过期时间
    """
    return int(getattr(settings, "CHAT_JOB_TTL_SECONDS", 86400))


def _done_ttl() -> int:
    """Job 终态后的过期时间：只需覆盖断线重连与迟到追更，远小于 running 期。"""
    return int(getattr(settings, "CHAT_JOB_DONE_TTL_SECONDS", 3600) or 3600)


def _hb_interval() -> float:
    """心跳刷新间隔（秒）。下限 1 秒，避免配成 0 时空转打 Redis。"""
    return max(1.0, float(getattr(settings, "CHAT_JOB_HEARTBEAT_INTERVAL_SECONDS", 5.0) or 5.0))


def _hb_ttl() -> int:
    """心跳键 TTL（秒）。下限 3 秒，且必须明显大于刷新间隔，否则 Redis 抖动就会误判。"""
    return max(3, int(getattr(settings, "CHAT_JOB_HEARTBEAT_TTL_SECONDS", 15) or 15))


def _single_instance() -> bool:
    """是否单实例部署。决定 boot_id 能否作为「即时判死」的依据。"""
    return bool(getattr(settings, "CHAT_JOB_SINGLE_INSTANCE", True))


def _is_job_alive(job_id: str, meta: dict[str, Any] | None = None) -> bool:
    """判断一个「声称 running」的 job 是否真的还有执行体在跑。

    为什么需要它：job 是进程内 asyncio task（见 create_chat_job 的 create_task），
    进程被杀或重启后 task 消失，但 Redis 里的 meta 仍带着 status=running 能活 24 小时。
    只看状态会让该会话被永久判定为「有任务在跑」：新建被 409、切分支被 409、
    SSE 追更永不退出。这里用「心跳 + boot_id」把 running 从静态状态变成有时效的断言。

    保守原则：读心跳抛异常时返回 True —— 宁可有最多一个 TTL 的延迟，
    也不误杀真正在跑的任务。
    """
    if meta is None:
        meta = get_job_meta(job_id)
    if not meta or meta.get("status") != "running":
        return False
    hb_state = cache.exists(_hb_key(job_id))
    if hb_state is None:
        # Redis 抖动：无法判断，按存活处理（宁可有延迟，也不误杀真正在跑的任务）。
        # 注意不能用 get_json —— 它把「键不存在」和「读失败」都返回 None，会区分不开。
        return True
    hb_present = hb_state
    same_boot = str(meta.get("boot_id") or "") == _BOOT_ID
    if hb_present:
        # 心跳还在跳 → 有执行体。单实例下若 boot_id 不同，说明是本进程重启前遗留的 job
        # （旧心跳最多还能活一个 TTL，这期间不该继续挡住用户）
        return same_boot if _single_instance() else True
    # 心跳缺失
    if _single_instance():
        # 本进程创建的 job 即使心跳写失败也应视为存活
        return same_boot
    return False


async def _heartbeat_loop(job_id: str) -> None:
    """周期性刷新心跳键，直到被取消。

    心跳是纯辅助信号，失败绝不能影响生成主流程，因此所有异常吞掉只记 debug。
    """
    interval = _hb_interval()
    ttl = _hb_ttl()
    while True:
        try:
            await asyncio.to_thread(
                cache.set_json,
                _hb_key(job_id),
                {"boot_id": _BOOT_ID, "at": time.time()},
                ttl,
            )
        except Exception:
            logger.debug("job {} 心跳写入失败（忽略，不影响生成）", job_id)
        await asyncio.sleep(interval)


async def _reap_dead_job(job_id: str, meta: dict[str, Any]) -> None:
    """把一个「声称 running 但没有执行体」的 job 收敛到终态。

    顺序有讲究：先写 error 事件再改状态。已连接的 SSE 是靠 status 变化退出的，
    若先改状态再写事件，那条说明原因的事件就永远读不到了。

    幂等：重复调用可能多写一条事件，但状态与锁操作都幂等，无副作用。
    """
    cancelled = False
    try:
        cancelled = await asyncio.to_thread(is_job_cancel_requested, job_id)
    except Exception:
        pass
    status = "cancelled" if cancelled else "failed"
    age = time.time() - float(meta.get("created_at") or 0)
    grace = max(0, int(getattr(settings, "CHAT_JOB_ORPHAN_GRACE_SECONDS", 5) or 0))
    if age >= grace:
        logger.warning("job {} 已无执行体（创建于 {:.0f}s 前），收敛为 {}", job_id, age, status)
    else:
        logger.info("job {} 已无执行体（创建仅 {:.1f}s，可能刚重启），收敛为 {}", job_id, age, status)
    if not cancelled:
        # 用户主动停止的任务不写错误事件，避免误导
        try:
            next_seq = await asyncio.to_thread(cache.llen, _events_key(job_id))
            await _append_event(job_id, next_seq, {"type": "error", "content": _ORPHAN_REASON})
        except Exception:
            logger.debug("job {} 孤儿收尾事件写入失败（不影响状态收敛）", job_id)
    try:
        meta["status"] = status
        meta["error"] = None if cancelled else _ORPHAN_REASON
        await asyncio.to_thread(cache.set_json, _meta_key(job_id), meta, _done_ttl())
        await asyncio.to_thread(cache.expire, _events_key(job_id), _done_ttl())
        await asyncio.to_thread(cache.delete, _cancel_key(job_id))
    except Exception:
        logger.warning("job {} 孤儿状态写入失败", job_id)
    # 仅当占用锁仍指向本 job 时才释放，防止误删新任务的锁
    await asyncio.to_thread(
        _release_active_key,
        int(meta.get("user_id", 0)),
        int(meta.get("agent_id", 0)),
        str(meta.get("session_id", "")),
        job_id,
    )


async def _append_event(job_id: str, seq: int, data: dict[str, Any]) -> None:
    """
    追加 Job 事件：RPUSH + meta/events 双 EXPIRE 经单次 Lua 往返完成。
    """
    wrapped = json.dumps({"seq": seq, "data": data}, ensure_ascii=False)
    await asyncio.to_thread(
        cache.append_event_atomic,
        _events_key(job_id),
        _meta_key(job_id),
        wrapped,
        _ttl(),
    )


async def create_chat_job(
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    message: str,
    use_knowledge_retrieval: bool,
    use_web_search: bool = False,
    attachment_ids: list[str] | None = None,
    regenerate: bool = False,
    target_message_id: int | None = None,
    mcp_approved_pending_id: str | None = None,
    resume_thread_id: str | None = None,
    resume_decision: str | None = None,
) -> tuple[str, bool]:
    """
    做三件事：并发互斥 → 写 meta → 起协程
    创建 Job：若同会话已有 running 任务则返回 (existing_job_id, True)。
    否则返回 (new_job_id, False)。
    """
    # 检查是否已有 running 任务
    # 如果已有 running 任务，则返回 (existing_job_id, True)
    # 否则返回 (new_job_id, False)
    ak = _active_key(user_id, agent_id, session_id)
    existing = await asyncio.to_thread(cache.get_json, ak)
    if isinstance(existing, dict) and existing.get("job_id"):
        ej = str(existing["job_id"])
        meta = await asyncio.to_thread(cache.get_json, _meta_key(ej))
        if meta and meta.get("status") == "running":
            # status 是 running 不代表还有执行体（进程可能已被重启），必须先验心跳。
            # 已死则就地收敛：写结束事件、置终态、释放锁，然后继续抢锁开新任务。
            if not await asyncio.to_thread(_is_job_alive, ej, meta):
                await _reap_dead_job(ej, meta)
            elif await asyncio.to_thread(is_job_cancel_requested, ej):
                await asyncio.to_thread(_release_active_key, user_id, agent_id, session_id, ej)
            else:
                return ej, True

    job_id = uuid.uuid4().hex
    # SET NX 抢锁（Redis 不可用返回 False）
    lock_ok = await asyncio.to_thread(cache.set_nx, ak, {"job_id": job_id}, _ttl())
    if not lock_ok:
        raced = await asyncio.to_thread(cache.get_json, ak)
        if isinstance(raced, dict) and raced.get("job_id"):
            return str(raced["job_id"]), True
        raise RuntimeError("无法创建对话任务：会话锁不可用")

    meta = {
        "job_id": job_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "session_id": session_id,
        "status": "running",
        "error": None,
        "regenerate": bool(regenerate),
        "target_message_id": target_message_id,
        # 创建进程标识与创建时刻：前者用于单实例下断定「上个进程遗留」，
        # 后者仅用于孤儿收敛时的日志分级
        "boot_id": _BOOT_ID,
        "created_at": time.time(),
    }
    await asyncio.to_thread(cache.set_json, _meta_key(job_id), meta, _ttl())

    # ★★★创建异步任务执行对话，后台协程入口
    aids = attachment_ids or []
    asyncio.create_task(
        _run_chat_job(
            job_id=job_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            message=message,
            use_knowledge_retrieval=use_knowledge_retrieval,
            use_web_search=use_web_search,
            attachment_ids=aids,
            regenerate=regenerate,
            target_message_id=target_message_id,
            mcp_approved_pending_id=mcp_approved_pending_id,
            resume_thread_id=resume_thread_id,
            resume_decision=resume_decision,
        )
    )
    return job_id, False


async def _run_chat_job(
    *,
    job_id: str,
    user_id: int,
    agent_id: int,
    session_id: str,
    message: str,
    use_knowledge_retrieval: bool,
    use_web_search: bool = False,
    attachment_ids: list[str] | None = None,
    regenerate: bool = False,
    target_message_id: int | None = None,
    mcp_approved_pending_id: str | None = None,
    resume_thread_id: str | None = None,
    resume_decision: str | None = None,
) -> None:
    """
    校验 → 排队 → 调用主流程 → 逐条事件写 Redis → 收尾改状态
    """
    from app.services.user_agent_recent import touch_recent_agent

    seq = 0
    # 快照累积状态：正文按事件逐块累加，达阈值后整体落到 mg_chat_turn_events。
    # 这里必须放在 try 之前：异常分支写终态快照时也要能拿到已生成的内容。
    text_parts: list[str] = []
    buf_chars = 0
    snapshot_at = time.monotonic()
    # 心跳必须在任何耗时工作之前启动：进程一旦被杀，心跳先停，
    # 重启后的存活判定才能立刻认出「这个 job 没有执行体」。
    hb_task = asyncio.create_task(_heartbeat_loop(job_id))
    try:
        # 获取智能体
        ua = await get_accessible_agent(agent_id, user_id)
        # 如果智能体不存在或无权限，则返回错误
        if not ua:
            await _append_event(job_id, seq, {"type": "error", "content": "智能体不存在或无权限"})
            seq += 1
            await _finish_meta(job_id, status="failed", error="智能体不存在")
            return

        user_cancelled = False
        gate = _llm_inflight()
        global _llm_inflight_waiting
        # 闸门已满时先给前端一条排队事件（含等待人数），避免 running 状态下长时间零反馈
        if gate.locked():
            await _append_event(job_id, seq, {"type": "queued", "waiting": _llm_inflight_waiting + 1})
            seq += 1
        _llm_inflight_waiting += 1
        acquired = False
        try:
            try:
                await asyncio.wait_for(gate.acquire(), timeout=_queue_timeout())
                acquired = True
            except asyncio.TimeoutError:
                await _append_event(job_id, seq, {"type": "error", "content": "服务繁忙，排队等待超时，请稍后重试"})
                seq += 1
                await _finish_meta(job_id, status="failed", error="排队等待超时")
                return
            # ★★★★★ 进入主流程
            async for ev in iter_chat_stream_events(
                ua,
                message,
                user_id,
                agent_id,
                session_id,
                use_knowledge_retrieval=use_knowledge_retrieval,
                use_web_search=use_web_search,
                attachment_ids=attachment_ids or [],
                regenerate=regenerate,
                target_message_id=target_message_id,
                cancel_check=lambda jid=job_id: is_job_cancel_requested(jid),
                mcp_approved_pending_id=mcp_approved_pending_id,
                # 审批恢复：thread 指原 job 的 checkpoint，decision 决定批准/拒绝
                resume_thread_id=resume_thread_id,
                resume_decision=resume_decision,
                # 传给工作区做容器命名，便于 cleanup 与残留清扫对账
                job_id=job_id,
            ):
                await _append_event(job_id, seq, ev)
                seq += 1
                if ev.get("type") == "content":
                    piece = ev.get("content") or ""
                    if piece:
                        text_parts.append(piece)
                        buf_chars += len(piece)
                # 字符数或时间任一达阈值即写快照（后台任务，不阻塞流式输出）
                if buf_chars and (
                    buf_chars >= _snapshot_chars()
                    or (time.monotonic() - snapshot_at) >= _snapshot_interval()
                ):
                    _spawn_snapshot(
                        job_id,
                        seq,
                        user_id=user_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        text_parts=text_parts,
                    )
                    buf_chars = 0
                    snapshot_at = time.monotonic()
                if ev.get("type") == "done" and ev.get("cancelled"):
                    user_cancelled = True
        finally:
            _llm_inflight_waiting -= 1
            if acquired:
                gate.release()

        if user_cancelled:
            await _finish_meta(job_id, status="cancelled", error=None)
            await _save_terminal_snapshot(
                job_id,
                seq,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                kind=KIND_CANCELLED,
                text_parts=text_parts,
            )
        else:
            await _finish_meta(job_id, status="completed", error=None)
            await _save_terminal_snapshot(
                job_id,
                seq,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                kind=KIND_DONE,
                text_parts=text_parts,
            )
            # 更新最近使用智能体（编辑器试聊会话不置顶）
            if not is_editor_preview_session(session_id):
                try:
                    await touch_recent_agent(user_id, agent_id)
                except Exception:
                    pass
    except Exception as e:
        await _append_event(job_id, seq, {"type": "error", "content": str(e)})
        await _finish_meta(job_id, status="failed", error=str(e))
        await _save_terminal_snapshot(
            job_id,
            seq,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            kind=KIND_ERROR,
            text_parts=text_parts,
            error=str(e),
        )
    finally:
        # 停心跳并删键：让后续判定立刻看到「无执行体」，不必等心跳 TTL 自然过期
        hb_task.cancel()
        try:
            await hb_task
        except (asyncio.CancelledError, Exception):
            pass
        try:
            await asyncio.to_thread(cache.delete, _hb_key(job_id))
        except Exception:
            pass
        # 仅当占用锁仍指向本 job 时释放：取消即时终结后可能已有新任务持有该锁
        await asyncio.to_thread(_release_active_key, user_id, agent_id, session_id, job_id)


async def _finish_meta(job_id: str, *, status: str, error: str | None) -> None:
    """
    完成任务：终态后改用较短的 done TTL，避免事件列表在 Redis 中留存 24h。
    """
    meta = await asyncio.to_thread(cache.get_json, _meta_key(job_id))
    if not isinstance(meta, dict):
        meta = {"job_id": job_id}
    meta["status"] = status
    meta["error"] = error
    ttl = _done_ttl()
    await asyncio.to_thread(cache.set_json, _meta_key(job_id), meta, ttl)
    await asyncio.to_thread(cache.expire, _events_key(job_id), ttl)
    await asyncio.to_thread(cache.delete, _cancel_key(job_id))


def get_job_meta(job_id: str) -> dict[str, Any] | None:
    """
    获取 Job 元数据
    """
    raw = cache.get_json(_meta_key(job_id))
    return raw if isinstance(raw, dict) else None


def _snapshot_chars() -> int:
    """累积多少字符触发一次快照。"""
    return max(200, int(getattr(settings, "CHAT_TURN_SNAPSHOT_CHARS", 2000) or 2000))


def _snapshot_interval() -> float:
    """距上次快照超过该秒数也触发一次，避免长停顿期间没有可恢复点。"""
    return max(0.1, float(getattr(settings, "CHAT_TURN_SNAPSHOT_INTERVAL_SECONDS", 0.5) or 0.5))


def _spawn_snapshot(
    job_id: str,
    seq: int,
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    text_parts: list[str],
) -> None:
    """后台写一条中途快照，不阻塞事件循环。

    这里刻意不 await：快照是尽力而为的兜底，让一次 DB 写拖慢流式输出是得不偿失的。
    `save_snapshot` 内部已吞掉所有异常，因此不会产生「任务异常未取回」的告警。
    """
    if not turn_events_enabled():
        return
    content = "".join(text_parts)
    if not content:
        return
    asyncio.create_task(
        asyncio.to_thread(
            save_snapshot,
            job_id=job_id,
            seq=seq,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            content=content,
        )
    )


async def _save_terminal_snapshot(
    job_id: str,
    seq: int,
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    kind: str,
    text_parts: list[str],
    error: str | None = None,
) -> None:
    """写终态快照并清掉中途快照。

    与中途快照不同，这里**刻意 await**：终态是断线恢复时唯一的完整依据，
    值得付一次 DB 往返的代价（收尾阶段一次，不影响流式体感）。
    """
    if not turn_events_enabled():
        return
    try:
        await asyncio.to_thread(
            save_terminal,
            job_id=job_id,
            seq=seq,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            kind=kind,
            content="".join(text_parts),
            error=error,
        )
    except Exception:
        logger.warning("终态快照写入失败 job_id={} kind={}", job_id, kind)


async def _resume_snapshot_line(job_id: str) -> str | None:
    """从 PG 取该 job 的最后快照，拼成一条 SSE 行；无快照或内容为空返回 None。

    只在 Redis 判定为不可用时调用，正常路径不会触碰数据库。
    """
    if not turn_events_enabled():
        return None
    snap = await asyncio.to_thread(load_latest_snapshot, job_id)
    if not snap:
        return None
    payload = snap.get("payload") or {}
    content = payload.get("content") or ""
    if not content:
        return None
    event = {
        "type": "resume_snapshot",
        "content": content,
        "seq": snap.get("seq", 0),
        "kind": snap.get("kind"),
    }
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def iter_job_sse_events(
    job_id: str,
    *,
    since_seq: int,
) -> Any:
    """
    异步迭代 SSE 行（不含外层 StreamingResponse），从 Redis 列表下标 since_seq 起追更直至任务结束。
    前端收流：从 Redis 列表下标 since_seq 起追更直至任务结束
    """
    # 从 Redis 列表下标 since_seq 起追更直至任务结束
    next_idx = max(0, since_seq)
    # 空闲退避：有事件时保持 40ms 低延迟，持续空转时逐倍放宽到 0.3s，降低挂起连接的 Redis 底噪
    idle_delay = 0.04
    # 上次校验执行体存活的时间（monotonic），配合 _LIVENESS_CHECK_INTERVAL 节流
    last_liveness_check = 0.0
    # 是否已经做过一次孤儿收敛：防止「收敛后状态没写进去」导致的反复收敛死循环
    reaped = False
    # 循环直到任务结束
    while True:
        # 批量取事件（下标即 seq，since_seq 续传语义不变）
        chunk = await asyncio.to_thread(cache.lrange_str, _events_key(job_id), next_idx, next_idx + 63)
        if chunk:
            idle_delay = 0.04
            for raw in chunk:
                ev_seq: Any = None
                data: dict[str, Any] | None = None
                try:
                    wrapped = json.loads(raw)
                    ev_seq = wrapped.get("seq")
                    data = wrapped.get("data") or {}
                except Exception:
                    data = None
                next_idx += 1
                # Redis 重启后事件列表从下标 0 重建，而 seq 继续递增，二者会错位。
                # 按事件体内的 seq 过滤，避免把已消费过的事件重发一遍（事件体本来就带 seq）。
                if isinstance(ev_seq, int) and ev_seq < since_seq:
                    continue
                if data is not None:
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            continue

        meta = await asyncio.to_thread(cache.get_json, _meta_key(job_id))
        if not meta:
            # Redis 里连 meta 都没了（重启或过期）：用 PG 快照收尾，
            # 否则客户端只会拿到一个空流，无从知道这轮生成到哪了。
            resume_line = await _resume_snapshot_line(job_id)
            if resume_line:
                yield resume_line
            break
        st = meta.get("status")
        if st != "running":
            break
        # 声称 running 但执行体已随进程消失时，本流必须收尾，否则前端永远转圈。
        # 校验按固定间隔节流：追更循环每圈只有 40ms，不能每圈都多读一次 Redis。
        now_mono = time.monotonic()
        if now_mono - last_liveness_check >= _LIVENESS_CHECK_INTERVAL:
            last_liveness_check = now_mono
            if not await asyncio.to_thread(_is_job_alive, job_id, meta):
                if reaped:
                    # 已收敛过一次仍未退出（例如状态写失败），直接结束，避免死循环
                    break
                reaped = True
                # 先收敛（写 error 事件 + 置终态），再 continue 一轮把刚写入的事件读出去；
                # 下一轮因 status 已是 failed 而自然退出。若此处直接 break，
                # 那条说明原因的事件就永远送不到已连接的客户端。
                await _reap_dead_job(job_id, meta)
                continue
            # 同一次节流窗口里顺带判定「事件列表是否己丢失」：
            # 客户端说消费过 since_seq 条，但 Redis 里可读条目不足以覆盖它的位置，
            # 说明列表被重建过（重启）或已过期 → 用 PG 快照收尾。
            # 用 llen < next_idx 而不是 llen == 0：列表重建后会从 0 重新增长，很快就不为 0。
            if since_seq > 0:
                llen = await asyncio.to_thread(cache.llen, _events_key(job_id))
                if llen < next_idx:
                    resume_line = await _resume_snapshot_line(job_id)
                    if resume_line:
                        yield resume_line
                    break
        # 短轮询：在 Job 仍 running 且暂无新事件时等待；间隔过大会导致 SSE 观感像“整段输出”
        await asyncio.sleep(idle_delay)
        idle_delay = min(idle_delay * 2, 0.3)

    yield "data: [DONE]\n\n"


def verify_job_owner(job_id: str, user_id: int) -> bool:
    """
    验证 Job 是否属于用户
    """
    # 获取 Job 元数据
    meta = get_job_meta(job_id)
    # 如果 Job 元数据不存在，则返回 False
    if not meta:
        return False
    # 如果 Job 元数据中的 user_id 与传入的 user_id 不匹配，则返回 False
    # 否则返回 True
    return int(meta.get("user_id", -1)) == int(user_id)
