"""测量 import 连带量：单独 import 某个模块后，会把多少 app.chat.* 拉起。

为什么单独一个脚本而不是直接写成 test：本轮的基线（改造前 37 个）需要在改代码
之前先记录一次，脚本可以随时对任意模块重复测量；测试文件只负责断言最终门槛。
用法：
    python scripts/measure_import_graph.py                 # 默认测 app.kb.rag_pipeline
    python scripts/measure_import_graph.py app.kb.search_tool
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def measure(module: str) -> dict:
    """在干净子进程里 import 目标模块，回报连带量。"""
    code = f"""
import json, sys
import {module}
chat = sorted(m for m in sys.modules if m.startswith("app.chat"))
print(json.dumps({{
    "chat_modules": chat,
    "chat_count": len(chat),
    "tortoise": "tortoise" in sys.modules,
    "fastapi": "fastapi" in sys.modules,
    "total_modules": len(sys.modules),
}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise SystemExit(f"测量失败：\n{proc.stderr}")
    # 只取最后一行，防止目标模块自己往 stdout 打印内容
    import json

    return json.loads(proc.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "app.kb.rag_pipeline"
    result = measure(target)
    print(f"import {target}")
    print(f"  app.chat.* 模块数：{result['chat_count']}")
    print(f"  tortoise 被拉起：{result['tortoise']}")
    print(f"  fastapi 被拉起： {result['fastapi']}")
    print(f"  sys.modules 总量：{result['total_modules']}")
    if result["chat_modules"]:
        print("  明细：")
        for m in result["chat_modules"]:
            print(f"    - {m}")
