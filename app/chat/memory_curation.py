"""U7 跨会话记忆沉淀：从会话对话里提炼「稳定的、跨会话仍然成立」的事实。

与会话级记忆（`memory_archive`）的分工：
- `memory_archive`：把被 compact 挤出窗口的轮次**原话**切块存 Milvus，补"本会话早期说过什么"
- 本模块：把若干轮对话**提炼成事实条目**存 PG（+ Milvus 索引），补"这个用户/这个智能体长期怎么做事"

两套互不依赖：archive 靠"离开原文窗口"触发（苛刻），curation 靠"攒够未提炼轮数"触发。

设计要点（每一条都是踩过的）：
1. **宁少勿多**。一条错记忆比十条缺记忆更伤——用户会失去信任，且错误记忆会被一直注入。
   prompt 里把"不该提炼"的类别写死，并要求模型没有可提炼的就输出 `[]`。
2. **按 `mem_key` 对账，不是追加**。把「已有记忆」一并喂给模型，让它知道哪些该更新。
3. **水位线只标记成功提炼的轮次**。LLM 失败时不推进，下次重试。
4. **PG 是真相源，Milvus 可落后**。双写失败只记 `embedding_synced=false`，由补扫兜底。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

from langchain_core.messages import HumanMessage

from app.chat.memory_store import (
    DEFAULT_MAX_ACTIVE,
    enforce_limit,
    list_active,
    list_unsynced,
    mark_synced,
    upsert_memory,
)
from app.settings import settings
from app.utils.llm_factory import LlmFactory

logger = logging.getLogger(__name__)

_META_CURATED_TURN_KEYS = "memory_curated_turn_keys"

CURATION_PROMPT = """你在维护一个 AI 助手的「长期记忆」。输入是用户与助手的若干轮对话，\
你要输出**跨会话仍然成立**的稳定事实。

## 应该提炼（满足任一即可）
- 用户的技术栈、工具偏好、开发环境约定（例：主力数据库、操作系统、常用命令风格）
- 用户明确表达的工作习惯或沟通偏好（例：结论先行、反感套话）
- 项目层面的约定与规则（例：改写重要文档前必须先备份）
- 用户已做出的技术决策及理由（例：选 PostgreSQL 因为需要 JSONB 索引）
- 长期有效的结论、判据、踩坑经验（例：本地 curl 必须加 --noproxy）

## 绝对不要提炼
- 一次性的数据查询结果、临时文件内容、具体某次报错
- 时效性内容（"明天再改""这次先这样""下次注意"）
- 助手自己的推测或建议（除非用户明确接受或采纳）
- 寒暄、确认、情绪表达（"好的""谢谢""没问题"）
- 与【已有记忆】重复且没有变化的内容

## 输出格式
只输出 JSON 数组，不要任何解释。没有值得记的就输出 []。
示例：
[{{"key": "primary_database", "text": "主力数据库是 MySQL 8.0.41，开发时不容器化，直连宿主机 3306", "kind": "fact", "confidence": 0.9}}]

## 字段要求
- key：英文小写下划线，稳定标识「这条记忆说的是哪件事」。
  ⚠️ **最重要的一条**：如果【已有记忆】里已经有描述同一件事的 key，**必须原样复用那个 key**，
  哪怕措辞不同、哪怕你觉得能起个更准的名字。换 key 会导致新旧两条同时生效，
  等于记忆里同时存在两个互相矛盾的版本。
  只有【已有记忆】里确实没有对应主题时，才新建 key。别用 v1/v2 这种带版本号的写法。
- text：一句中文陈述，用第三人称描述用户或项目（不要出现"你""用户说"）
- kind：fact | preference | convention | decision
- confidence：0~1。用户明确说过的给 0.9，助手建议且用户默许的给 0.6

最多输出 {max_items} 条，按重要性从高到低排序。

## 已有记忆（格式：key: text）
> 这些 key 已经存在。本次若发现语义相同的事，请**直接沿用对应 key**，让新值覆盖旧值。

{existing}

## 待提炼的对话
{turns}
"""


def memory_user_scope(user_id: int, agent_id: int) -> str:
    """用户级记忆的隔离键：`u{user}_a{agent}_mem`。

    为什么带 agent：不同智能体用途可能完全无关（写代码的 vs 答疑的），
    记忆串了会很怪。也与会话级 `u{a}_s{sid}` 的前缀自然衔接。
    """
    return f"u{int(user_id)}_a{int(agent_id)}_mem"


def curation_enabled() -> bool:
    return bool(getattr(settings, "CHAT_MEMORY_CURATION_ENABLED", True))


def user_scope_enabled() -> bool:
    return curation_enabled() and bool(getattr(settings, "CHAT_MEMORY_USER_SCOPE_ENABLED", True))


def _max_items() -> int:
    return max(1, int(getattr(settings, "CHAT_MEMORY_CURATION_MAX_ITEMS", 6) or 6))


def _min_turns() -> int:
    return max(1, int(getattr(settings, "CHAT_MEMORY_CURATION_MIN_TURNS", 5) or 5))


# ---------------------------------------------------------------------------
# JSON 解析：LLM 输出不可靠，必须容错
# ---------------------------------------------------------------------------


def parse_items(raw: str, *, max_items: int | None = None) -> list[dict[str, Any]]:
    """从模型输出里解析条目数组。任何异常都返回空列表，绝不抛。

    为什么要这么宽容：模型有时会在 JSON 前后加解释、用 ```json 围栏、或者
    把整个数组包在对象里。解析失败就当作"这次没提炼出东西"，比让整个后台任务
    崩掉好——提炼本来就是尽力而为。
    """
    text = (raw or "").strip()
    if not text:
        return []
    # 去掉 markdown 围栏
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []

    limit = _max_items() if max_items is None else max(1, int(max_items))
    out: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        body = str(item.get("text") or "").strip()
        if not key or not body:
            continue
        # key 规范化：模型可能给出中文、空格、大写
        key = re.sub(r"[^a-z0-9_]+", "_", key.lower()).strip("_")[:60]
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        kind = str(item.get("kind") or "fact").strip().lower()
        if kind not in {"fact", "preference", "convention", "decision"}:
            kind = "fact"
        try:
            conf = float(item.get("confidence", 0.8))
        except (TypeError, ValueError):
            conf = 0.8
        out.append(
            {
                "key": key,
                "text": body[:500],
                "kind": kind,
                "confidence": max(0.0, min(1.0, conf)),
            }
        )
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# 提炼
# ---------------------------------------------------------------------------


def _turns_text_for(turn_items: list[list[Any]]) -> str:
    """把若干轮 LangChain 消息转成提炼用的纯文本。"""
    from app.chat.memory_archive import turn_to_text

    parts: list[str] = []
    for idx, turn in enumerate(turn_items, 1):
        body = turn_to_text(turn)
        if body:
            parts.append(f"### 第 {idx} 轮\n{body}")
    return "\n\n".join(parts)


def _existing_memory_block(user_id: int, agent_id: int, *, limit: int = 60) -> str:
    rows = list_active(user_id, agent_id, limit=limit)
    if not rows:
        return "（暂无）"
    return "\n".join(f"- {r['mem_key']}: {r['text']}" for r in rows)


def extract_items(
    turns_text: str,
    llm_config: dict[str, Any],
    *,
    existing_block: str = "（暂无）",
) -> list[dict[str, Any]]:
    """调 LLM 提炼条目。任何失败返回空列表。"""
    body = (turns_text or "").strip()
    if not body:
        return []
    if not (llm_config.get("api_key") or "").strip():
        logger.debug("extract_items: skip (no api_key)")
        return []
    prompt = CURATION_PROMPT.format(
        max_items=_max_items(),
        existing=existing_block,
        turns=body[:24000],  # 防超长：只喂最近的对话内容
    )
    try:
        model = LlmFactory.try_chat(llm_config)
        if model is None:
            return []
        out = model.invoke([HumanMessage(content=prompt)])
        raw = getattr(out, "content", None) or str(out)
        items = parse_items(str(raw))
        logger.info("记忆提炼：候选 %s 条", len(items))
        return items
    except Exception:
        logger.exception("extract_items failed")
        return []


# ---------------------------------------------------------------------------
# 双写 Milvus
# ---------------------------------------------------------------------------


def sync_to_milvus(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """把 PG 里待同步的条目写进 / 移出 Milvus。

    :return: (写入条数, 删除条数)
    """
    if not rows:
        return 0, 0
    from app.chat.milvus_memory import get_chat_memory_milvus
    from app.kb.multimodal_embedding import get_multimodal_embedding_service

    active = [r for r in rows if r.get("status") == "active"]
    retired = [r for r in rows if r.get("status") != "active"]

    mgr = get_chat_memory_milvus()
    try:
        mgr.init_collection()
    except Exception:
        logger.exception("sync_to_milvus: init_collection failed")
        return 0, 0

    deleted = 0
    if retired:
        try:
            deleted = mgr.delete_by_chunk_ids([f"memitem_{r['id']}" for r in retired])
            mark_synced([r["id"] for r in retired])
        except Exception:
            logger.exception("sync_to_milvus: delete retired failed")

    inserted = 0
    if active:
        try:
            embedder = get_multimodal_embedding_service()
            texts = [r["text"] for r in active]
            vectors = embedder.get_text_embeddings(texts)
            payload = []
            for r, vec in zip(active, vectors):
                payload.append(
                    {
                        "memory_scope": memory_user_scope(r["user_id"], r["agent_id"]),
                        "text": r["text"][:8192],
                        # 会话级才用得上这三个字段；用户级条目恒 0，但 schema 要求必填
                        "turn_index": 0,
                        "turn_key": 0,
                        "chunk_index": 0,
                        "chunk_id": f"memitem_{r['id']}",
                        "dense_embedding": vec,
                    }
                )
            mgr.insert(payload)
            mark_synced([r["id"] for r in active])
            inserted = len(payload)
        except Exception:
            logger.exception("sync_to_milvus: insert active failed")
    return inserted, deleted


def flush_unsynced(*, limit: int = 100) -> tuple[int, int]:
    """补扫：把 `embedding_synced=false` 的条目同步到 Milvus。"""
    rows = list_unsynced(limit=limit)
    if not rows:
        return 0, 0
    return sync_to_milvus(rows)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _curated_keys(meta: dict | None) -> set[int]:
    return {int(k) for k in ((meta or {}).get(_META_CURATED_TURN_KEYS) or [])}


def pending_turns_for_curation(
    user_id: int, agent_id: int, session_id: str
) -> tuple[list[list[Any]], list[int]]:
    """返回 (未提炼的轮次消息, 这些轮次的 turn_key)。

    与 `memory_archive.archived_turn_keys_on_path` 是两套水位线：
    archive 记 `memory_archived_turn_keys`（已写进 Milvus 的原话），
    curation 记 `memory_curated_turn_keys`（已提炼成条目）。互不影响。
    """
    from app.chat.storage import storage
    from app.core.memory_turns import group_turn_pairs, split_system_prefix

    pairs = storage.load_path_with_ids(user_id, agent_id, session_id)
    if not pairs:
        return [], []
    msgs = [m for _, m in pairs]
    _, body = split_system_prefix(msgs)
    body_pairs = pairs[len(pairs) - len(body) :] if body else []
    turn_pairs = group_turn_pairs(body_pairs)
    if not turn_pairs:
        return [], []

    meta = storage.get_session_metadata(user_id, agent_id, session_id)
    done = _curated_keys(meta)
    pending: list[list[Any]] = []
    keys: list[int] = []
    for tp in turn_pairs:
        key = tp[0][0]
        if key in done:
            continue
        pending.append([m for _, m in tp])
        keys.append(int(key))
    return pending, keys


def curate_session_memories(
    user_id: int,
    agent_id: int,
    session_id: str,
    llm_config: dict[str, Any],
) -> dict[str, Any]:
    """提炼一次。返回统计信息（便于日志与验收）。"""
    stats: dict[str, Any] = {
        "enabled": curation_enabled(),
        "turns": 0,
        "candidates": 0,
        "created": 0,
        "updated": 0,
        "unchanged": 0,
        "milvus_inserted": 0,
        "milvus_deleted": 0,
        "skipped": None,
    }
    if not stats["enabled"]:
        stats["skipped"] = "开关关闭"
        return stats

    pending, keys = pending_turns_for_curation(user_id, agent_id, session_id)
    stats["turns"] = len(pending)
    if len(pending) < _min_turns():
        stats["skipped"] = f"未提炼轮次 {len(pending)} < 阈值 {_min_turns()}"
        return stats

    turns_text = _turns_text_for(pending)
    items = extract_items(
        turns_text,
        llm_config,
        existing_block=_existing_memory_block(user_id, agent_id),
    )
    stats["candidates"] = len(items)

    touched_ids: list[int] = []
    for item in items:
        res = upsert_memory(
            user_id,
            agent_id,
            mem_key=item["key"],
            text=item["text"],
            kind=item["kind"],
            confidence=item["confidence"],
            source_session=session_id,
            source_turn_key=keys[0] if keys else None,
        )
        action = res.get("action")
        if action == "created":
            stats["created"] += 1
            touched_ids.append(int(res["id"]))
        elif action == "updated":
            stats["updated"] += 1
            touched_ids.append(int(res["id"]))
            if res.get("superseded_id"):
                touched_ids.append(int(res["superseded_id"]))
        elif action == "unchanged":
            stats["unchanged"] += 1

    # 超上限淘汰最低 confidence 的
    evicted = enforce_limit(
        user_id,
        agent_id,
        max_active=int(getattr(settings, "CHAT_MEMORY_CURATION_MAX_ACTIVE", DEFAULT_MAX_ACTIVE) or DEFAULT_MAX_ACTIVE),
    )
    touched_ids.extend(evicted)

    if touched_ids:
        rows = [r for r in list_unsynced(limit=200) if r["id"] in set(touched_ids)]
        ins, dele = sync_to_milvus(rows)
        stats["milvus_inserted"] = ins
        stats["milvus_deleted"] = dele

    # 只有提炼真的成功过（有候选或有 untouched）才推进水位线，
    # 否则模型挂了会导致这批轮次被永久跳过
    if items or stats["skipped"] is None:
        from app.chat.storage import storage

        meta = storage.get_session_metadata(user_id, agent_id, session_id)
        merged = sorted(_curated_keys(meta) | {int(k) for k in keys})
        storage.patch_session_metadata(
            user_id,
            agent_id,
            session_id,
            {_META_CURATED_TURN_KEYS: merged},
        )
    logger.info("记忆提炼完成：%s", stats)
    return stats


def schedule_curate_session_memories(
    user_id: int,
    agent_id: int,
    session_id: str,
    llm_config: dict[str, Any],
) -> None:
    """对话落库后调用：按配置同步或后台线程提炼。"""
    if not curation_enabled():
        return

    def _run() -> None:
        try:
            curate_session_memories(user_id, agent_id, session_id, llm_config)
        except Exception:
            logger.exception("schedule_curate_session_memories")

    if getattr(settings, "CHAT_MEMORY_CURATION_ASYNC", True):
        threading.Thread(target=_run, name="chat-memory-curation", daemon=True).start()
    else:
        _run()


def purge_memories_for_agent(user_id: int, agent_id: int) -> int:
    """删智能体时清理它的全部记忆（含 Milvus 向量）。"""
    from app.chat.memory_store import purge_agent_memories
    from app.chat.milvus_memory import get_chat_memory_milvus

    ids = purge_agent_memories(user_id, agent_id)
    if ids:
        try:
            mgr = get_chat_memory_milvus()
            mgr.init_collection()
            mgr.delete_by_chunk_ids([f"memitem_{i}" for i in ids])
        except Exception:
            logger.exception("purge_memories_for_agent: milvus 清理失败（PG 已删）")
    return len(ids)
