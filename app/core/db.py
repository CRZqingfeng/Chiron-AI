"""
PostgreSQL（业务库）SQLAlchemy 引擎与会话工厂。

位置说明：从 `app/chat/database.py` 迁到 `app/core/db.py`。它管理的是**全业务域**
共用的 Base / engine / SessionLocal——对话五表与知识库三表都挂在这同一个 Base 上，
所以它不该住在 `app.chat` 下面（那会让 `app.kb` 必须 import `app.chat`）。

建表约定：`create_all` 只创建「已 import 并注册到 Base.metadata」的表。
因此 `init_db()` 里显式 import 两个模型模块——漏了就会静默少建表。
KB 专属的 ALTER 语句放在 `app/kb/db_migrations.py`，由本函数回调。
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.settings import settings

os.environ.setdefault("PGCLIENTENCODING", "UTF8")


def _sqlalchemy_database_url(raw: str) -> str:
    """
    优先使用 psycopg3；兼容 .env 中的 postgresql+psycopg2://。
    """
    for old, new in (
        ("postgresql+psycopg2://", "postgresql+psycopg://"),
        ("postgres+psycopg2://", "postgresql+psycopg://"),
    ):
        if raw.startswith(old):
            return new + raw[len(old) :]
    return raw


def _make_engine():
    """
    创建SQLAlchemy引擎, 用于连接聊天 postgres 数据库
    """
    url = _sqlalchemy_database_url(settings.chat_database_url)
    return create_engine(url, pool_pre_ping=True)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

# 创建SQLAlchemy基类, 用于创建聊天数据库模型
Base = declarative_base()  


def init_db() -> None:
    """
    让 ORM 模型挂到 Base.metadata 上
    在数据库里创建缺失的表（对话五表 + 知识库三表）
    """
    # 关键：create_all 只建「已注册到 Base.metadata」的表。
    # 这两个 import 就是注册动作，缺一个就会静默少建表。
    import app.core.db_models  # noqa: F401
    import app.kb.db_models  # noqa: F401

    from sqlalchemy import text

    Base.metadata.create_all(bind=engine)
    try:
        from app.chat.milvus_memory import init_chat_memory_collection

        init_chat_memory_collection()
    except Exception:
        import logging

        logging.getLogger(__name__).warning("init_chat_memory_collection skipped", exc_info=True)
    # create_all 不会为已有表添加新列，需显式迁移
    with engine.begin() as conn:
        conn.execute(
            text("ALTER TABLE mg_chat_messages ADD COLUMN IF NOT EXISTS rag_steps JSON")
        )
        conn.execute(
            text("ALTER TABLE mg_chat_messages ADD COLUMN IF NOT EXISTS content_json JSON")
        )
        conn.execute(text("ALTER TABLE mg_chat_messages ADD COLUMN IF NOT EXISTS error_text TEXT"))
        conn.execute(text("ALTER TABLE mg_chat_messages ADD COLUMN IF NOT EXISTS sources JSON"))
        conn.execute(text("ALTER TABLE mg_chat_messages ADD COLUMN IF NOT EXISTS thinking_text TEXT"))
        conn.execute(text("ALTER TABLE mg_chat_messages ADD COLUMN IF NOT EXISTS thinking_items JSON"))
        # 审批中断：这条助手消息是「暂停等用户决定」的中间态，resume 到手后原地覆盖
        conn.execute(
            text("ALTER TABLE mg_chat_messages ADD COLUMN IF NOT EXISTS approval_pending JSON")
        )
        # 消息树：parent_id / selected_child_id（存量线性会话回填为单链树）
        conn.execute(
            text("ALTER TABLE mg_chat_messages ADD COLUMN IF NOT EXISTS parent_id INTEGER")
        )
        conn.execute(
            text("ALTER TABLE mg_chat_messages ADD COLUMN IF NOT EXISTS selected_child_id INTEGER")
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_mg_chat_messages_parent_id "
                "ON mg_chat_messages (parent_id)"
            )
        )
        # 回填 parent_id：仅处理仍为空且存在前驱行的存量线性数据；
        # 树模式下非根消息插入时必然已带 parent_id，不会被误改
        conn.execute(
            text(
                """
                WITH ordered AS (
                    SELECT id,
                           LAG(id) OVER (PARTITION BY session_ref_id ORDER BY id) AS prev_id
                    FROM mg_chat_messages
                )
                UPDATE mg_chat_messages m
                SET parent_id = o.prev_id
                FROM ordered o
                WHERE m.id = o.id
                  AND m.parent_id IS NULL
                  AND o.prev_id IS NOT NULL
                """
            )
        )
        # 回填 selected_child_id：仅处理仍为空且确有子行的存量数据（线性时每行至多一个子行）
        conn.execute(
            text(
                """
                UPDATE mg_chat_messages p
                SET selected_child_id = (
                    SELECT MAX(c.id) FROM mg_chat_messages c WHERE c.parent_id = p.id
                )
                WHERE p.selected_child_id IS NULL
                  AND EXISTS (SELECT 1 FROM mg_chat_messages c WHERE c.parent_id = p.id)
                """
            )
        )
        conn.execute(
            text(
                "ALTER TABLE mg_chat_sessions ADD COLUMN IF NOT EXISTS last_user_preview VARCHAR(160)"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE mg_chat_sessions ADD COLUMN IF NOT EXISTS path_message_count INTEGER"
            )
        )

    # 知识库表的迁移放回知识库自己的包里，core 不写业务 DDL
    from app.kb.db_migrations import migrate_kb_schema

    migrate_kb_schema(engine)


def get_db_session() -> Session:
    """
    获取聊天数据库会话
    """
    return SessionLocal()
