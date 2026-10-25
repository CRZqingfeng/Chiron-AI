"""跨会话记忆的查询与删除（U7）。

只做两件事：列出该智能体下生效中的记忆、删除某一条。
**不做编辑**——编辑等价于「删一条 + 写一条」，但写要走 curation 的对账逻辑
（保证 mem_key 唯一、旧值标 superseded），随手给个 PUT 会绕过那套约束。

为什么删除必须走这里而不是直接删库：删 PG 行前要先删 Milvus 里的向量
（`chunk_id = memitem_{pg_id}`），行删了就拼不出 chunk_id 了。
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query

from app.chat.memory_curation import memory_user_scope
from app.chat.memory_store import get_item, hard_delete, list_active
from app.core.dependency import DependAuth
from app.models.admin import User
from app.schemas.base import Fail, Success
from app.services.user_agent_access import get_accessible_agent

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/memories", summary="列出跨会话记忆", tags=["智能体模块"])
async def list_memories(
    agent_id: int = Query(..., description="智能体 ID"),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = DependAuth,
):
    """列出该智能体下**生效中**（status=active）的记忆条目。

    只返回 active 的：superseded 是历史版本，给用户看"你以前记的是 MySQL"没有意义，
    但库里有留存——需要追溯时走数据库。
    """
    user_id = int(current_user.id)
    ua = await get_accessible_agent(agent_id, user_id)
    if not ua:
        return Fail(code=404, msg="智能体不存在或无权限访问")
    rows = await asyncio.to_thread(list_active, user_id, agent_id, limit=limit)
    return Success(
        data={
            "items": rows,
            "total": len(rows),
            "scope": memory_user_scope(user_id, agent_id),
        }
    )


@router.delete("/memories/{mem_id}", summary="删除一条跨会话记忆", tags=["智能体模块"])
async def delete_memory(
    mem_id: int,
    agent_id: int = Query(..., description="智能体 ID"),
    current_user: User = DependAuth,
):
    """删除一条记忆（PG 行 + Milvus 向量）。

    顺序不能反：先删 Milvus（要拿 mem_id 拼 chunk_id），再删 PG 行。
    反了的话行没了、id 也拿不到，Milvus 里会留一条永远召不回来但占位的记录。
    """
    user_id = int(current_user.id)
    ua = await get_accessible_agent(agent_id, user_id)
    if not ua:
        return Fail(code=404, msg="智能体不存在或无权限访问")

    row = await asyncio.to_thread(get_item, user_id, agent_id, mem_id)
    if not row:
        return Fail(code=404, msg="记忆不存在")

    try:
        from app.chat.milvus_memory import get_chat_memory_milvus

        mgr = get_chat_memory_milvus()
        await asyncio.to_thread(mgr.init_collection)
        await asyncio.to_thread(mgr.delete_by_chunk_ids, [f"memitem_{mem_id}"])
    except Exception:
        # Milvus 删失败不该阻止用户删除：PG 是真相源，检索侧会用 status 过滤掉
        # （`retrieve_user_memory_hits` 命中后会回 PG 校验 status=='active'）
        logger.exception("delete_memory: milvus 删除失败（继续删 PG）mem_id=%s", mem_id)

    ok = await asyncio.to_thread(hard_delete, user_id, agent_id, mem_id)
    if not ok:
        return Fail(code=500, msg="删除失败")
    return Success(msg="已删除")
