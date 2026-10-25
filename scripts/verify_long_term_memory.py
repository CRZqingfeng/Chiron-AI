"""U7 端到端验收：跨会话记忆沉淀。

验的不是"代码看起来对"，而是四件必须用真实数据回答的事：
- E1 curation 真的产出了条目（`mg_chat_memories`）
- E2 条目真的进了 Milvus（能语义召回）
- E3 **跨会话**真的生效（新会话里能查到旧会话的事实）← 唯一能证明"沉淀成功"的
- E4 同 `mem_key` 会覆盖而不是追加（用户改主意时不能两条并存）

全程用独立测试会话，跑完自动清理。
"""

from __future__ import annotations

import sys

sys.path.insert(0, r"D:\develop\develop\code\GitHub-Project\Chiron-AI-main\Chiron-AI-main")

import uuid

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import text as sql

from app.chat.memory_curation import (
    curate_session_memories,
    memory_user_scope,
    purge_memories_for_agent,
)
from app.chat.memory_search import retrieve_user_memory_hits
from app.chat.memory_store import list_active
from app.chat.storage import storage
from app.core.db import SessionLocal
from app.utils.api_key_crypto import decrypt_api_key_safe

USER_ID, AGENT_ID = 1, 2
TAG = uuid.uuid4().hex[:8]
SESSION_A = f"__u7_e2e_a_{TAG}__"
SESSION_B = f"__u7_e2e_b_{TAG}__"


def _pad(text: str, target: int = 800) -> str:
    while len(text) < target:
        text += "\n" + text
    return text[:target]


# 会话 A：明确说出 3 个稳定事实
TURNS_A: list[tuple[str, str]] = [
    ("我主力数据库用的是 MySQL 8.0.41，开发时不容器化，直连宿主机 3306", _pad("明白，记下这个环境约定。")),
    ("我习惯用 Git Bash，命令喜欢用 && 连成单行", _pad("收到，以后给命令就按这个风格。")),
    ("改写重要文档之前必须先备份，这个约定别忘了", _pad("好，这条我记住了。")),
    ("今天天气不错", _pad("确实不错。")),
    ("帮我看下这段代码有没有问题", _pad("这段看起来没问题。")),
    ("好的谢谢", _pad("不客气。")),
]

# 会话 A 追加的 5 轮：第 1 轮改主意，其余是噪音。
# 为什么要凑够 5 轮：`CHAT_MEMORY_CURATION_MIN_TURNS=5`，只加 1 轮会 skipped，
# 那样「旧值新值不并存」是假通过——根本没提炼，当然不会并存。
TURNS_CHANGE: list[tuple[str, str]] = [
    (
        "我决定把数据库换成 PostgreSQL 了，因为需要 JSONB 索引",
        _pad("了解，那之前关于 MySQL 的约定要更新。"),
    ),
    ("迁移时要注意什么", _pad("注意 JSON 列要改成 JSONB，键顺序会变。")),
    ("索引怎么建", _pad("按查询模式建，别一把梭。")),
    ("连接池配多大", _pad("先按并发量的两倍起步。")),
    ("好，先这样", _pad("行。")),
]


def _llm_config() -> dict:
    db = SessionLocal()
    try:
        row = db.execute(
            sql("select api_key_ciphertext, base_url, model_name from user_agent where id = :i"),
            {"i": AGENT_ID},
        ).fetchone()
    finally:
        db.close()
    if not row:
        raise SystemExit("智能体不存在")
    return {
        "api_key": (decrypt_api_key_safe(row[0]) or "").strip(),
        "base_url": (row[1] or "").strip() or None,
        "model_name": row[2],
    }


def _write_turns(session: str, turns: list[tuple[str, str]]) -> None:
    msgs = []
    for human, ai in turns:
        msgs.append(HumanMessage(content=human))
        msgs.append(AIMessage(content=ai))
    storage.append_messages(USER_ID, AGENT_ID, session, msgs)


def _cleanup() -> None:
    db = SessionLocal()
    try:
        for s in (SESSION_A, SESSION_B):
            db.execute(
                sql(
                    "delete from mg_chat_messages where session_ref_id = "
                    "(select id from mg_chat_sessions where session_id = :s)"
                ),
                {"s": s},
            )
            db.execute(sql("delete from mg_chat_sessions where session_id = :s"), {"s": s})
        db.commit()
    finally:
        db.close()


def main() -> int:
    llm = _llm_config()
    print("=" * 72)
    print("U7 端到端验收：跨会话记忆沉淀")
    print(f"  智能体 LLM: {llm['model_name']} | 测试标记: {TAG}")
    print("=" * 72)

    # 起跑前清掉该智能体的旧记忆，避免污染判定
    removed = purge_memories_for_agent(USER_ID, AGENT_ID)
    print(f"① 清空该智能体旧记忆：{removed} 条")

    print(f"\n② 造会话 A（{len(TURNS_A)} 轮，含 3 个稳定事实）")
    _write_turns(SESSION_A, TURNS_A)
    print(f"   session = {SESSION_A}")

    print("\n③ 触发 curation（真实 LLM 提炼）")
    stats = curate_session_memories(USER_ID, AGENT_ID, SESSION_A, llm)
    print(f"   轮次={stats['turns']} 候选={stats['candidates']} 新建={stats['created']} "
          f"更新={stats['updated']} 不变={stats['unchanged']} "
          f"milvus写入={stats['milvus_inserted']} milvus删除={stats['milvus_deleted']}")
    if stats.get("skipped"):
        print(f"   跳过原因: {stats['skipped']}")

    print("\n④ E1：查 PG 里的 active 条目")
    rows = list_active(USER_ID, AGENT_ID, limit=50)
    print(f"   active 条数: {len(rows)}")
    for r in rows:
        print(f"   - [{r['kind']}] {r['mem_key']}: {r['text'][:52]} (conf={r['confidence']})")
    e1 = len(rows) > 0

    print("\n⑤ E2：查 Milvus 里的用户级向量")
    scope = memory_user_scope(USER_ID, AGENT_ID)
    from app.chat.milvus_memory import get_chat_memory_milvus

    mgr = get_chat_memory_milvus()
    mgr.init_collection()
    client = mgr._get_client()
    try:
        client.flush(mgr.collection_name)
    except Exception:  # noqa: BLE001
        pass
    vec_rows = client.query(
        mgr.collection_name,
        filter=f'memory_scope == "{scope}"',
        output_fields=["text", "chunk_id"],
        limit=100,
    )
    print(f"   scope={scope} 行数={len(vec_rows)}")
    e2 = len(vec_rows) >= len(rows) and len(vec_rows) > 0

    print(f"\n⑥ E3：在**另一个会话** B 里检索（{SESSION_B}）")
    _write_turns(SESSION_B, [("你好", _pad("你好，有什么可以帮你？", 300))])
    for q in ("我用什么数据库", "我习惯用什么终端", "改文档要注意什么"):
        hits = retrieve_user_memory_hits(q, user_id=USER_ID, agent_id=AGENT_ID, top_k=3)
        src = hits[0]["source"] if hits else "-"
        top = hits[0]["text"][:44] if hits else "（无命中）"
        print(f"   问「{q}」→ {len(hits)} 条（来源={src}）: {top}")
    hits_db = retrieve_user_memory_hits("我用什么数据库", user_id=USER_ID, agent_id=AGENT_ID, top_k=3)
    e3 = any("MySQL" in h["text"] for h in hits_db)

    print("\n⑦ E4：用户改主意（MySQL → PostgreSQL），再提炼一次")
    _write_turns(SESSION_A, TURNS_CHANGE)
    stats2 = curate_session_memories(USER_ID, AGENT_ID, SESSION_A, llm)
    print(f"   轮次={stats2['turns']} 候选={stats2['candidates']} 新建={stats2['created']} "
          f"更新={stats2['updated']} 不变={stats2['unchanged']} skipped={stats2.get('skipped')}")
    rows2 = list_active(USER_ID, AGENT_ID, limit=50)
    db_keys = [r["mem_key"] for r in rows2]
    print(f"   active 条数: {len(rows2)}（键: {db_keys}）")
    db_rows = [r for r in rows2 if "数据库" in r["text"] or "PostgreSQL" in r["text"] or "MySQL" in r["text"]]
    for r in db_rows:
        print(f"   - {r['mem_key']}: {r['text'][:60]}")
    # 判据按 mem_key 精确判定。不能拿"文本里有没有 MySQL"当判据 —— 新值本身就可能
    # 写成"已从 MySQL 换成 PostgreSQL"，那句里也有 MySQL，会误判成旧值还在。
    db_items = [r for r in rows2 if r["mem_key"] == "primary_database"]
    updated_to_pg = len(db_items) == 1 and "PostgreSQL" in db_items[0]["text"]
    old_still_active = any(("PostgreSQL" not in r["text"]) for r in db_items)
    # key 漂移检测：已有主题的 key 个数不应超过首次提炼的 key 数（换 key 会让条目越攒越多）
    keys_drifted = len(rows2) > len(rows)
    e4 = bool(stats2["candidates"]) and updated_to_pg and not old_still_active and not keys_drifted
    print(f"   判据：有候选={bool(stats2['candidates'])} primary_database条数={len(db_items)} "
          f"已更新为PG={updated_to_pg} 旧值仍active={old_still_active} "
          f"key发生漂移={keys_drifted}（{len(rows)}→{len(rows2)} 条）")

    print("\n⑧ E5：关开关应完全回退")
    from unittest import mock

    from app.settings import settings

    with mock.patch.object(settings, "CHAT_MEMORY_CURATION_ENABLED", False):
        stats3 = curate_session_memories(USER_ID, AGENT_ID, SESSION_A, llm)
        hits_off = retrieve_user_memory_hits("我用什么数据库", user_id=USER_ID, agent_id=AGENT_ID, top_k=3)
    print(f"   提炼 skipped={stats3['skipped']} | 检索命中={len(hits_off)}")
    e5 = bool(stats3["skipped"]) and not hits_off

    print("\n⑨ 清理")
    _cleanup()
    purge_memories_for_agent(USER_ID, AGENT_ID)
    try:
        mgr.delete_by_scope(scope)
    except Exception:  # noqa: BLE001
        pass
    print("   已删测试会话、记忆条目与向量")

    print("\n" + "=" * 72)
    for name, ok in (("E1 产出条目", e1), ("E2 写入 Milvus", e2), ("E3 跨会话生效", e3),
                     ("E4 同 key 覆盖", e4), ("E5 开关回退", e5)):
        print(f"  {'✅' if ok else '❌'} {name}")
    all_ok = all([e1, e2, e3, e4, e5])
    print("=" * 72)
    print("结论：" + ("✅ 跨会话记忆沉淀跑通" if all_ok else "❌ 有验收项未通过"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
