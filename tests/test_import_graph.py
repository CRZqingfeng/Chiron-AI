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


# ---------------------------------------------------------------------------
# 分层依赖守卫（D-046）
#
# 目标依赖方向：api → chat → kb → core → utils
#   - core 是地基，不能知道任何业务域（chat / kb / mcp_client / controllers）
#   - chat 是执行链路，不能回头依赖 controllers / api（那会让「跑一轮对话」
#     必须先把 HTTP 层的模块树拉起来）
#   - services 是业务能力的落脚点，同样不能依赖 controllers
#
# 这些规则以前只靠人看，2026-10-25 体检时发现 core 有 3 处、chat 有 2 处反向依赖。
# 现在把它们钉成断言：新增一行就越界变红。
# ---------------------------------------------------------------------------

_FORBIDDEN_UPWARD = {
    "app/core": ("app.chat", "app.kb", "app.mcp_client", "app.controllers", "app.api"),
    "app/chat": ("app.controllers", "app.api"),
    "app/services": ("app.controllers", "app.api"),
}


def _scan_upward_imports(package: str, forbidden: tuple[str, ...]) -> set[tuple[str, str]]:
    """返回 (文件, 被依赖的上层模块) 集合。只看代码行，跳过 docstring。"""
    import ast

    root = PROJECT_ROOT / package
    hits: set[tuple[str, str]] = set()
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for bad in forbidden:
                        if alias.name == bad or alias.name.startswith(bad + "."):
                            hits.add((path.relative_to(PROJECT_ROOT).as_posix(), bad))
                continue
            if not module:
                continue
            for bad in forbidden:
                if module == bad or module.startswith(bad + "."):
                    hits.add((path.relative_to(PROJECT_ROOT).as_posix(), bad))
    return hits


class LayerDirectionTests(TestCase):
    def test_core_does_not_depend_on_business_packages(self) -> None:
        """core 只能往下（utils/settings/log），不能知道 chat/kb/mcp/controllers。"""
        hits = _scan_upward_imports("app/core", _FORBIDDEN_UPWARD["app/core"])
        self.assertEqual(
            hits,
            set(),
            f"app/core 出现向上依赖（应由组合根 app/main.py 或 app/init_app.py 承担装配）：{sorted(hits)}",
        )

    def test_chat_does_not_depend_on_http_layers(self) -> None:
        """对话执行链路不能依赖 controllers/api：它要能在无 HTTP 上下文的场景跑（job/脚本）。"""
        hits = _scan_upward_imports("app/chat", _FORBIDDEN_UPWARD["app/chat"])
        self.assertEqual(hits, set(), f"app/chat 反向依赖 HTTP 层：{sorted(hits)}")

    def test_services_do_not_depend_on_http_layers(self) -> None:
        hits = _scan_upward_imports("app/services", _FORBIDDEN_UPWARD["app/services"])
        self.assertEqual(hits, set(), f"app/services 反向依赖 HTTP 层：{sorted(hits)}")

    def test_init_db_only_touches_own_domain(self) -> None:
        """init_db 不再顺手建 KB 表 / 起记忆 Milvus —— 那两件归组合根。"""
        src = (PROJECT_ROOT / "app" / "core" / "db.py").read_text(encoding="utf-8")
        self.assertNotIn("from app.kb.db_migrations import", src)
        self.assertNotIn("from app.chat.milvus_memory import", src)
        # 组合根必须补上这两步，否则启动时静默少建表 / 记忆集合不预热
        main_src = (PROJECT_ROOT / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn("migrate_kb_schema", main_src)
        self.assertIn("init_chat_memory_collection", main_src)
        self.assertIn("import app.kb.db_models", main_src)

    def test_business_patches_are_registered_by_bootstrap(self) -> None:
        """业务 schema 补丁靠业务包自己登记，组合根必须 import 那个模块。

        这里要分两层断言，缺一不可：
        ① 登记机制本身可用（import 业务 patches 后 all_patches() 能看到）；
        ② 组合根确实 import 了它 —— 因为登记是 **import 副作用**，
           删掉 `app/init_app.py` 里那一行，补丁会静默不执行、且不报错。
        """
        import app.mcp_client.patches  # noqa: F401 - 触发登记

        from app.core.schema_patches import all_patches

        names = {p.name for p in all_patches()}
        self.assertIn("100_mcp_preset_confirm_policy", names, "登记机制失效")

        init_src = (PROJECT_ROOT / "app" / "init_app.py").read_text(encoding="utf-8")
        self.assertIn(
            "import app.mcp_client.patches",
            init_src,
            "组合根没有 import 业务补丁模块，启动时该补丁会被静默跳过",
        )
        core_src = (PROJECT_ROOT / "app" / "core" / "schema_patches.py").read_text(encoding="utf-8")
        self.assertNotIn("from app.mcp_client", core_src, "core 不该再直接读 MCP 预置")
