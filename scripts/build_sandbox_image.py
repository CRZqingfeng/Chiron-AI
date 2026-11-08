"""构建工作区沙箱镜像（U10 P1-B）。

用法：
    python scripts/build_sandbox_image.py [--tag chiron-sandbox:py311] [--no-cache]

为什么要脚本化：镜像里装的包与 `command_policy._SAFE_BINARIES` 是一份**契约**，
手工敲 docker build 容易忘参数、也容易拿错上下文。真正的对齐校验在
`tests/workspace/test_sandbox_image.py`，这里只负责构建。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "deploy" / "sandbox.Dockerfile"
# 上下文用 deploy/：仓库根带 venv/ 与 node_modules/，传上下文能传几分钟
BUILD_CONTEXT = REPO_ROOT / "deploy"
DEFAULT_TAG = "chiron-sandbox:py311"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建工作区沙箱镜像")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"镜像标签（默认 {DEFAULT_TAG}）")
    parser.add_argument("--no-cache", action="store_true", help="不使用构建缓存")
    args = parser.parse_args(argv)

    if not DOCKERFILE.exists():
        print(f"找不到 Dockerfile：{DOCKERFILE}", file=sys.stderr)
        return 2

    cmd = ["docker", "build", "-f", str(DOCKERFILE), "-t", args.tag]
    if args.no_cache:
        cmd.append("--no-cache")
    cmd.append(str(BUILD_CONTEXT))

    print("$ " + " ".join(cmd))
    code = subprocess.call(cmd)
    if code == 0:
        print(f"\n构建完成：{args.tag}")
        print(f"启用方式：在 .env 里设 CHAT_WORKSPACE_IMAGE={args.tag}")
        print("（默认值刻意不改：镜像缺失时会静默降级到本地目录，没有隔离，比用 slim 更糟）")
    return code


if __name__ == "__main__":
    sys.exit(main())
