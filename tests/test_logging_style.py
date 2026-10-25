"""日志占位符纪律：loguru 只认 `{}`，写 `%s` 会把参数静默丢掉。

为什么值得一条测试守着：这个 bug 不报错、不告警，只是**把原因从日志里抹掉**。
2026-10-24 体检时它直接造成一次误诊——checkpoint 降级为内存，日志只剩字面量 `%s`，
真实原因（psycopg 不认 ProactorEventLoop）被吞，于是被误判成「依赖没装」。

判定方式：文件里用了 loguru 的 logger（`from loguru import logger` 或
`from app.log import logger`），却把含 `%s/%d/%r/%f` 的字符串当首参传给
`logger.xxx(...)` —— 一律视为 bug。标准库 logging 不在此列（它走 InterceptHandler
时 `record.getMessage()` 会正确插值）。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest import TestCase

APP_DIR = Path(__file__).resolve().parent.parent / "app"

# 会进 loguru 的日志方法
_LOG_METHODS = {
    "trace",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "exception",
    "critical",
}

# % 风格占位符；不用 %%，也不用 "100%" 这类普通百分号
_PERCENT_PLACEHOLDER = re.compile(r"%[-+ #0]*\d*(?:\.\d+)?[sdfr]")


class LoggingStyleTests(TestCase):
    def test_loguru_calls_use_brace_placeholder(self) -> None:
        offenders: list[str] = []
        for path in sorted(APP_DIR.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if (
                "from loguru import logger" not in source
                and "from app.log import logger" not in source
            ):
                continue
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute) or func.attr not in _LOG_METHODS:
                    continue
                if not (isinstance(func.value, ast.Name) and func.value.id == "logger"):
                    continue
                first = node.args[0]
                # f-string / 变量 / 非字面量都跳过，只看写死的模板串
                if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                    continue
                if _PERCENT_PLACEHOLDER.search(first.value):
                    rel = path.relative_to(APP_DIR.parent).as_posix()
                    offenders.append(f"{rel}:{node.lineno}")

        self.assertEqual(
            offenders,
            [],
            "以下 logger 调用用了 % 风格占位符，运行时参数会被丢掉：\n  "
            + "\n  ".join(offenders),
        )
