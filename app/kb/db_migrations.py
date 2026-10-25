"""知识库表的增量迁移（create_all 不会给已有表加列）。

为什么单独一个文件：`app/core/db.py` 是全业务域共用的建表入口，不该写任何
单一业务域的 DDL。KB 的表结构演进放在 KB 自己的包里，由 `init_db()` 回调。

调用约定：由 `app.core.db.init_db()` 在建表之后调用，传入已创建的 engine，
本模块不自己持有连接（测试里换 sqlite 时能跟着换）。
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# 每条语句都必须幂等：init_db 在每次进程启动时都会跑一遍
_KB_MIGRATIONS = (
    # 文件内容 sha256：同内容重传时跳过重建向量
    "ALTER TABLE mg_kb_documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)",
)


def migrate_kb_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        for stmt in _KB_MIGRATIONS:
            conn.execute(text(stmt))
