# 工作区沙箱镜像（U10 P1-B）
#
# 为什么需要自制：命令白名单承诺 `git` / `tree` / `file` 等 17 个命令**免确认**执行，
# 而官方 `python:3.11-slim` 里连 git、tree、file 都没有。结果是「免审批通过、
# 容器里 command not found」，模型会反复重试、换写法，用户看到的现象是「审批坏了」。
#
# 镜像必须配得上白名单的承诺——这是本文件存在的全部理由。
#
# 构建：
#   python scripts/build_sandbox_image.py
#   （等价于 docker build -f deploy/sandbox.Dockerfile -t chiron-sandbox:py311 deploy）
#
# 启用：
#   .env 里设 CHAT_WORKSPACE_IMAGE=chiron-sandbox:py311
#   ⚠️ 默认值**不**改：镜像不存在时 DockerWorkspace 会静默降级到本地目录（无隔离），
#      那比用 slim 更糟。所以要不要用新镜像，必须是一个显式决定。
#
# 上下文用 deploy/ 而不是仓库根：仓库根带 venv/ 与 node_modules/，
# 构建上下文能有几个 GB，光是传上下文就要好几分钟。

FROM python:3.11-slim

# 换国内源再装包。实测直连 deb.debian.org 只有 ~37 kB/s，装 25MB 的 git 要等十几分钟；
# 换源后是秒级。这一步不是「优化」，是没有它就没法用——每次重建都要等。
# 用 deb822 的 .sources 文件（Debian 12+ 的默认格式），老格式的 sources.list 里没有内容。
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' \
            /etc/apt/sources.list.d/debian.sources; \
    else \
        sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' \
            /etc/apt/sources.list; \
    fi

# 1) 补齐白名单里 slim 缺的三个命令
# 2) procps 提供 ps——排查「进程还在不在」时要用，体积很小
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        tree \
        file \
        procps \
    && rm -rf /var/lib/apt/lists/*

# 内置 pytest：沙箱是 `--network none` 的，跑的时候装不了包，
# 所以测试依赖必须烧进镜像，否则「脚本级验证」只能靠手写 assert
RUN pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "pytest>=8,<9"

WORKDIR /workspace

# 自检：构建时就把「白名单承诺 vs 镜像现实」对齐，而不是等运行时才发现。
# 这份命令清单必须与 app/workspace/command_policy.py::_SAFE_BINARIES 一致——
# tests/workspace/test_sandbox_image.py 会再核一遍（镜像存在时才跑）。
RUN set -e; \
    for b in ls dir pwd cat head tail wc grep stat file du tree sort uniq cut diff git; do \
        command -v "$b" >/dev/null || { echo "MISSING: $b"; exit 1; }; \
    done; \
    python -c "import pytest; print('pytest', pytest.__version__)"; \
    echo "sandbox image self-check OK"

# 保持 root：非 root 需要宿主 uid 映射，Windows / Docker Desktop 下的挂载写权限
# 会变得不可控，而写文件能力是 P0 刚打通的主线。等 P2 做「容器按 session 复用」
# 时连同 uid 映射一起处理。
CMD ["/bin/bash"]
