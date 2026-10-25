"""结构断言：`app/kb` 不能反向依赖 `app/chat`（U8 的验收门槛）。

为什么用 subprocess 而不是直接 import：Python 的模块缓存是进程级的，
一个进程里 import 过一次 chat，后续测量就永远是「已加载」。要证明「干净进程里
import 检索模块不会拉起对话域」，只能在子进程里测。

两个硬指标（不是"感觉解耦了"）：
- `import app.kb.rag_pipeline` 后 `app.chat.*` 模块数 == 0
- 同时 FastAPI 应用与 Tortoise 都没被拉起（说明 `app/__init__.py` 不再是传染源）

改造前基线分别是 37 / True / True。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import TestCase

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 允许保留的反向依赖：以图检索要读对话域的附件表，属 U8b（图源抽象）的范围。
# 出现其它文件即说明有人新加了一条 `app/kb → app.chat`，必须当场发现。
_KB_CHAT_ALLOWLIST = {
    "app/kb/image_search_tool.py",
    "app/kb/rag_image_pipeline.py",
}


def _measure(module: str) -> dict:
    """在干净子进程里 import 目标模块，回报它连带拉起了什么。"""
    code = f"""
import gc, json, sys
import {module}

def has_fastapi_app():
    try:
        from fastapi import FastAPI
    except ImportError:
        return False
    return any(isinstance(o, FastAPI) for o in gc.get_objects())

print(json.dumps({{
    "chat_modules": sorted(m for m in sys.modules if m.startswith("app.chat")),
    "tortoise": "tortoise" in sys.modules,
    "fastapi_app": has_fastapi_app(),
}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise AssertionError(f"测量失败（import {module}）：\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


class ImportGraphTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 一次子进程覆盖两个断言，省掉重复的启动开销
        cls.rag_import = _measure("app.kb.rag_pipeline")

    def test_kb_import_pulls_no_chat_module(self) -> None:
        self.assertEqual(
            self.rag_import["chat_modules"],
            [],
            "import app.kb.rag_pipeline 拉起了对话域模块，app/kb 仍有反向依赖",
        )

    def test_kb_import_does_not_build_app(self) -> None:
        self.assertFalse(
            self.rag_import["fastapi_app"],
            "import 检索模块时创建了 FastAPI 实例：app/__init__.py 又在模块级装配应用了",
        )
        self.assertFalse(
            self.rag_import["tortoise"],
            "import 检索模块时拉起了 Tortoise：入口应从 app.main 导入，不要回到 app 包",
        )

    def test_no_new_reverse_dependency_from_kb(self) -> None:
        offenders = set()
        for path in (PROJECT_ROOT / "app" / "kb").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "from app.chat" in text or "import app.chat" in text:
                offenders.add(path.relative_to(PROJECT_ROOT).as_posix())
        self.assertEqual(
            offenders - _KB_CHAT_ALLOWLIST,
            set(),
            "app/kb 新增了对 app.chat 的依赖；检索模块必须能独立 import",
        )
        self.assertEqual(
            _KB_CHAT_ALLOWLIST - offenders,
            set(),
            "白名单里的文件已不再依赖 app.chat，请从白名单移除（别让豁免腐化）",
        )

    def test_shared_base_registers_both_domains(self) -> None:
        """create_all 只建「已注册到 Base.metadata」的表，漏注册就静默少建表。"""
        from app.core import db
        import app.core.db_models  # noqa: F401
        import app.kb.db_models  # noqa: F401

        expected = {
            # 对话域五表
            "mg_chat_sessions",
            "mg_chat_messages",
            "mg_chat_attachments",
            "mg_chat_memory_cursor",
            "mg_chat_turn_events",
            # 知识库三表
            "mg_kb_documents",
            "mg_kb_parent_chunks",
            "mg_kb_images",
        }
        self.assertEqual(expected - set(db.Base.metadata.tables), set())
