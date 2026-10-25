"""U7 第 0 步：验证现有「会话记忆」归档链路真能跑通。

为什么必须单独验：18 个会话的 metadata 全空、Milvus 记忆集合 0 行 —— 这条链路
一次都没在真实数据上跑过。不验它，U7 的用户级记忆就建在未验证的地基上，
端到端验收时"会话级"和"用户级"两个变量一起动，等于没验。

做法（全程用独立测试会话，不碰任何真实会话）：
1. 灌 8 轮假对话（约 20k 字符，够触发压缩）
2. 调 build_compacted_model_messages（阈值临时调低）→ 应写出 compact 状态
3. 调 archive_session_memory → 应写入 Milvus
4. 调 search_session_memory → 应能召回
5. 清理：删测试会话消息 + 删 Milvus 里的 scope

同时输出「按真实阈值 80000 需要多长会话才会触发」的估算，用来校准配置。
"""

from __future__ import annotations

import sys

sys.path.insert(0, r"D:\develop\develop\code\GitHub-Project\Chiron-AI-main\Chiron-AI-main")

import json
import uuid
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import text as sql

from app.chat.storage import storage
from app.core.db import SessionLocal
from app.settings import settings

USER_ID, AGENT_ID = 1, 2
SESSION = f"__u7_probe_{uuid.uuid4().hex[:8]}__"

# 每轮目标字符数：必须超过 keep_chars，否则「可压缩空间」为 0（_choose_keep_from 会返回 0）
_TURN_TARGET = 2300


def _pad(text: str, target: int = _TURN_TARGET) -> str:
    """把一段有信息量的话循环撑到目标长度，模拟真实的长回答。"""
    if len(text) >= target:
        return text[:target]
    filler = text
    while len(text) < target:
        text += "\n" + filler
    return text[:target]

# 8 轮对话，每轮 human 短、ai 长（模拟真实：助手回答比提问长得多）
TURNS_RAW: list[tuple[str, str]] = [
    (
        "我准备把项目的 MySQL 换掉，你觉得选哪个",
        "从你的使用强度看，PostgreSQL 更合适。理由是：1) 你有大量 JSON 字段（"
        "thinking_items、approval_pending、metadata_json），PG 的 JSONB 支持索引；"
        "2) 你已经在用它存 checkpoint；3) 单机场景下运维成本差不多。"
        "但要注意：迁移时 JSON 类型不能直接照搬，JSONB 会改变键顺序。"
    ),
    (
        "那向量检索怎么办",
        "两个选择：pgvector 装在同一个 PG 里，或者继续用 Milvus。"
        "你已经有 Milvus 且在跑知识库，继续用它更省事；pgvector 的好处是少一个组件，"
        "但需要换镜像（当前 PG 17.11 镜像里没有 vector 扩展，我实测过 pg_available_extensions 里查不到）。"
    ),
    (
        "改写文档前要备份吗",
        "要。你之前就因为直接覆盖导致丢过一次内容。约定：改重要文档前先 cp 备份，"
        "文件名带日期。这条我记下来，以后每次都提醒你。"
    ),
    (
        "我习惯用什么终端",
        "Git Bash，而且命令喜欢用 && 连成单行。你在 Windows 上遇到过引号嵌套崩溃的问题，"
        "所以复杂命令建议写成脚本文件。"
    ),
    (
        "测试跑不起来怎么办",
        "先分清是环境问题还是代码问题。你踩过的坑：venv 用 ./venv、没装 pytest 要用 unittest、"
        "临时端口撞 Windows 保留段要换端口、curl 本地要加 --noproxy。"
    ),
    (
        "我大四了该准备什么",
        "你 2027-06 毕业，目标 AI 全栈。当前最有价值的动作：把 Chiron-AI 这个项目做深，"
        "而不是再开新项目。面试时能讲清「为什么这么设计」比「做过什么」更重要。"
    ),
    (
        "知识库检索效果怎么样",
        "你的 RAG 是三级分块 + auto-merging + hybrid（dense+BM25）+ rerank。"
        "实测一次「多表查询案例」命中 5 段、3/5 相关、带来源页码。"
    ),
    (
        "今天就到这",
        "好。今天完成的事记一下：还完 4 项 P2 债、core 依赖归零、350 测试通过。"
        "下一步是 U7 记忆沉淀。"
    ),
]


TURNS: list[tuple[str, str]] = [(h, _pad(a)) for h, a in TURNS_RAW]


def _estimate_trigger_chars() -> None:
    """按真实阈值估算：会话要多长才会触发压缩。"""
    trigger = int(getattr(settings, "CHAT_COMPACT_TRIGGER_CHARS", 80000) or 80000)
    keep = int(getattr(settings, "CHAT_COMPACT_KEEP_CHARS", 24000) or 24000)
    tool_est = int(getattr(settings, "CHAT_COMPACT_TOOLS_ESTIMATE_CHARS", 8000) or 8000)
    headroom = int(getattr(settings, "CHAT_COMPACT_HEADROOM_CHARS", 12000) or 12000)
    print(f"  当前配置：trigger={trigger} keep={keep} tools_est={tool_est} headroom={headroom}")
    # 粗略：可压缩空间 = trigger - (tools + headroom + keep)  → 需要正文超过这个才触发
    room = trigger - tool_est - headroom - keep
    print(f"  正文需超过约 {max(0, room):,} 字符才会触发压缩（不含 system/摘要）")
    avg_turn = 2500
    print(f"  按每轮约 {avg_turn} 字符估：需要 {max(1, room // avg_turn)} 轮以上")


def main() -> int:
    print("=" * 70)
    print("U7 第 0 步：会话记忆归档链路验证")
    print("=" * 70)
    print("① 配置与触发门槛估算")
    _estimate_trigger_chars()

    print(f"\n② 灌入 8 轮假对话到测试会话 {SESSION}")
    msgs = []
    for human, ai in TURNS:
        msgs.append(HumanMessage(content=human))
        msgs.append(AIMessage(content=ai))
    storage.append_messages(USER_ID, AGENT_ID, SESSION, msgs)
    pairs = storage.load_path_with_ids(USER_ID, AGENT_ID, SESSION)
    total_chars = sum(len(str(getattr(m, "content", ""))) for _, m in pairs)
    print(f"  已写入 {len(pairs)} 条消息 / 正文 {total_chars:,} 字符")

    print("\n③ 压低阈值跑一次压缩（trigger=6000 / keep=2500，只为验证链路，不改真实配置）")
    from app.chat.compact import build_compacted_model_messages

    # 不走 Tortoise ORM：独立脚本里 Tortoise 没初始化，filter() 会抛
    # ConfigurationError: default_connection cannot be None。直接查库更省事。
    from app.utils.api_key_crypto import decrypt_api_key_safe

    db0 = SessionLocal()
    try:
        row = db0.execute(
            sql(
                "select api_key_ciphertext, base_url, model_name from user_agent where id = :i"
            ),
            {"i": AGENT_ID},
        ).fetchone()
    finally:
        db0.close()
    if not row:
        print("  ❌ 智能体不存在")
        return 1
    llm_cfg = {
        "api_key": (decrypt_api_key_safe(row[0]) or "").strip(),
        "base_url": (row[1] or "").strip() or None,
        "model_name": row[2],
    }
    print(f"  智能体 LLM: model={llm_cfg['model_name']} key={'有' if llm_cfg['api_key'] else '无'}")

    raw_messages = [m for _, m in pairs]
    with (
        mock.patch.object(settings, "CHAT_COMPACT_TRIGGER_CHARS", 6000),
        mock.patch.object(settings, "CHAT_COMPACT_KEEP_CHARS", 2500),
    ):
        view = build_compacted_model_messages(
            raw_messages,
            user_id=USER_ID,
            agent_id=AGENT_ID,
            session_id=SESSION,
            llm_config=llm_cfg,
            system_chars=1500,
            path_ids=[i for i, _ in pairs],
        )
    print(f"  压缩视图：{len(raw_messages)} 条 → {len(view)} 条")

    meta = storage.get_session_metadata(USER_ID, AGENT_ID, SESSION)
    states = meta.get("compact_states") or []
    summary = meta.get("compact_summary")
    print(f"  metadata keys: {list(meta.keys())}")
    print(f"  compact_states 条数: {len(states)}")
    if states:
        cov = states[-1].get("covered_turn_keys") or []
        print(f"    最近一条 covered 轮次: {cov}")
        print(f"    摘要长度: {len(states[-1].get('summary') or '')}")
    if summary:
        print(f"  compact_summary 长度: {len(summary)}")
    if not states and not summary:
        print("  ❌ 压缩未生效（没写出任何 compact 状态）")

    print("\n④ 触发归档 archive_session_memory")
    from app.chat.memory_archive import archive_session_memory

    archive_session_memory(USER_ID, AGENT_ID, SESSION)
    meta2 = storage.get_session_metadata(USER_ID, AGENT_ID, SESSION)
    archived_keys = meta2.get("memory_archived_turn_keys") or []
    print(f"  memory_archived_turn_keys: {archived_keys}")

    from app.chat.memory_scope import memory_scope_for

    scope = memory_scope_for(USER_ID, AGENT_ID, SESSION)
    from app.chat.milvus_memory import get_chat_memory_milvus

    mgr = get_chat_memory_milvus()
    mgr.init_collection()
    client = mgr._get_client()
    # Milvus 是最终一致的：insert 后必须 flush 才能被 query 看到。
    # 不 flush 会得到「行数 0 但 search 能召回」这种自相矛盾的结果（第一版就是）。
    try:
        client.flush(mgr.collection_name)
    except Exception as e:  # noqa: BLE001
        print(f"  flush 失败（继续用 search 判定）: {e}")
    rows = client.query(
        mgr.collection_name,
        filter=f'memory_scope == "{scope}"',
        output_fields=["text", "turn_key"],
        limit=50,
    )
    print(f"  Milvus 该 scope 行数: {len(rows)}")
    print(f"  scope = {scope}")
    if rows:
        print(f"    样例: {rows[0]['text'][:60]}…")

    print("\n⑤ 调 search_session_memory 验证召回")
    from app.chat.memory_search import search_session_memory

    text, trace = search_session_memory(
        "我要换数据库，之前建议选哪个",
        user_id=USER_ID,
        agent_id=AGENT_ID,
        session_id=SESSION,
        llm_config=llm_cfg,
    )
    print(f"  重写查询: {trace.get('rewritten_query')}")
    print(f"  命中数: {trace.get('hit_count')}")
    print(f"  返回前 120 字: {text[:120]}…")

    print("\n⑥ 清理测试数据")
    db = SessionLocal()
    try:
        db.execute(
            sql(
                "delete from mg_chat_messages where session_ref_id = "
                "(select id from mg_chat_sessions where session_id = :s)"
            ),
            {"s": SESSION},
        )
        db.execute(sql("delete from mg_chat_sessions where session_id = :s"), {"s": SESSION})
        db.commit()
        print("  已删测试会话与消息")
    finally:
        db.close()
    try:
        mgr.delete_by_scope(scope)
        print("  已删 Milvus 测试 scope")
    except Exception as e:  # noqa: BLE001
        print(f"  删 Milvus scope 失败（可忽略）: {e}")

    print("\n" + "=" * 70)
    # 判定以「search 能召回」为准：flush 的可见性依赖 Milvus 版本，
    # 而 search 走的是真实检索路径，才是用户实际看到的。
    ok = bool(trace.get("hit_count")) and bool(archived_keys)
    print("结论：" + ("✅ 归档与召回链路跑通" if ok else "❌ 链路仍有断点，见上面输出"))
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
