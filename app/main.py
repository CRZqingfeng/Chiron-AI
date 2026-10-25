"""ASGI 应用装配入口。

从 `app/__init__.py` 迁出：包初始化不应该顺手建一个 FastAPI 应用。
`app/__init__.py` 之前第 98 行的 `app = create_app()` 会在任何 `import app.xxx`
时执行，把 FastAPI、全部路由、整个 chat 包一起拉起来——这让 `app/kb` 的检索
代码无法被评测脚本单独 import。

启动方式：`uvicorn app.main:app`（`run.py` 已同步）。
`app/__init__.py` 现在只有 docstring，导入它不会产生任何副作用。
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from tortoise import Tortoise

from app.core.exceptions import SettingNotFound
from app.core.init_app import (
    init_data,
    make_middlewares,
    register_exceptions,
    register_routers,
)

try:
    from app.settings.config import settings
except ImportError:
    raise SettingNotFound("Can not import settings")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.log import logger

    await init_data()
    try:
        from app.core.object_storage import ensure_bucket

        # 对象存储 bucket 初始化（头像/会话附件/知识库文档与图片）；失败时文件功能不可用，其余功能照常
        await asyncio.to_thread(ensure_bucket)
    except Exception as e:
        logger.error("对象存储初始化失败（头像/附件/知识库文件功能将不可用）: {}", e)
    try:
        from app.core.db import init_db

        init_db()
        chat_db_ready = True
    except Exception as e:
        chat_db_ready = False
        logger.error("PostgreSQL 聊天库初始化失败（智能体对话将不可用）: {}", e)
    # 对话快照表的周期清理：只在建表成功后才起。
    # 表不存在时启动清理只会每轮刷一条警告，没有意义。
    cleanup_task: asyncio.Task | None = None
    if chat_db_ready:
        try:
            from app.chat.turn_events import run_cleanup_loop

            cleanup_task = asyncio.create_task(run_cleanup_loop())
        except Exception as e:
            logger.warning("对话快照清理任务启动失败（不影响主流程）: {}", e)
    try:
        from app.kb.milvus_client import MilvusManager

        # 预热知识库 Milvus 集合：首次连接较慢，提前到启动期避免首个上传任务等待
        await asyncio.to_thread(MilvusManager().init_collection)
    except Exception as e:
        logger.warning("Milvus 集合预热失败（首次上传时自动重试）: {}", e)
    if settings.DEBUG:
        logger.warning(
            "DEBUG=true：Header token=dev 可跳过 JWT，仅限本机调试，公网务必关闭"
        )
    yield
    # 先停清理任务：它可能在 await sleep，cancel 后无需额外清理资源
    if cleanup_task is not None:
        cleanup_task.cancel()
    try:
        from app.chat.checkpoint import close_checkpointer

        # 释放 checkpoint 连接池：不关会留下 psycopg 后台连接，进程退出时报警告
        await close_checkpointer()
    except Exception as e:
        logger.warning("关闭 checkpoint 连接池失败（不影响退出）: {}", e)
    try:
        from app.utils.egress import close_pinned_llm_clients

        await close_pinned_llm_clients()
    except Exception as e:
        logger.warning("关闭缓存的 LLM 客户端失败（不影响退出）: {}", e)
    await Tortoise.close_connections()


def create_app() -> FastAPI:
    docs_on = bool(getattr(settings, "DOCS_ENABLED", True))
    app = FastAPI(
        title=settings.APP_TITLE,
        description=settings.APP_DESCRIPTION,
        version=settings.VERSION,
        openapi_url="/openapi.json" if docs_on else None,
        docs_url="/docs" if docs_on else None,
        redoc_url="/redoc" if docs_on else None,
        middleware=make_middlewares(),
        lifespan=lifespan,
    )
    register_exceptions(app)
    register_routers(app, prefix="/api")
    return app


app = create_app()
