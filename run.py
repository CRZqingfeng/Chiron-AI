import asyncio
import sys

import uvicorn

from app.settings import settings

if __name__ == "__main__":
    # Windows 必须切到 SelectorEventLoop：psycopg 的 async 连接池在 ProactorEventLoop
    # 下直接拒绝工作（"Psycopg cannot use the 'ProactorEventLoop'"），而 langgraph 的
    # checkpoint 持久化依赖它。Linux 默认就是 Selector，无影响。
    # 必须在 uvicorn 创建事件循环之前设置，所以放这里而不是 app 包内。
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    host = (getattr(settings, "UVICORN_HOST", None) or "127.0.0.1").strip() or "127.0.0.1"
    # log_config=None：不用 uvicorn 自带 logging 配置（避免其 dictConfig 覆盖根处理器），
    # 标准库日志经 app.log 的 InterceptHandler 统一汇入 loguru
    uvicorn.run(
        # "app.main:app" 而不是 "app:app"：app/__init__.py 已瘦身成空壳，
        # ASGI 应用对象移到了 app/main.py（详见该文件顶部注释）
        "app.main:app",
        host=host,
        port=int(getattr(settings, "UVICORN_PORT", 19999)),
        reload=bool(settings.DEBUG),
        log_config=None,
    )
