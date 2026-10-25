"""Chiron-AI 应用包。

这里**故意不做任何事**——没有 `create_app()`，没有模块级 `app` 对象。

原因：Python 的 `import app.kb.rag_pipeline` 会先执行本文件。历史上这里第 98 行
是 `app = create_app()`，等于任何一次 import 都顺手装配整个 FastAPI 应用
（连带 FastAPI / 全部路由 / 整个 chat 包，实测 37 个 `app.chat.*` 模块），
让检索逻辑无法被评测脚本单独调用。

ASGI 入口已迁至 `app.main:app`。
"""
