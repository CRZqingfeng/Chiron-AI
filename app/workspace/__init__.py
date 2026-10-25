"""工作区层：为 agent 提供受控的文件读写与命令执行能力。

对外入口是 `open_workspace()`：
- 读配置决定后端（docker / local）
- 按 `{uid}/{aid}/{sid}` 分目录，做多租户隔离
- Docker 不可用时自动降级 local 并记 warning（对话不能因为沙箱不可用而中断）

`execute` 已实现但不作为工具暴露——审批中断（U5b）落地之前，模型不该具备
执行任意 shell 的能力。详见 `docs/dev/U4-思路.md` 发现二。
"""

from __future__ import annotations

import re
from pathlib import Path

from app.log import logger
from app.settings import settings
from app.workspace.base import (
    BaseWorkspace,
    CommandResult,
    FileInfo,
    PathEscapeError,
    WorkspaceError,
    WorkspaceUnavailable,
)
from app.workspace.local import LocalWorkspace
from app.workspace.tools import make_workspace_tools

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9_-]")


def _safe_segment(raw: str, fallback: str = "default") -> str:
    """把用户可控的标识转成安全的目录名。

    `session_id` 来自前端请求，直接拼进路径会形成目录穿越（`../../etc`）。
    这里只保留 `[A-Za-z0-9_-]`，其余一律替换为 `_`。
    """
    cleaned = _SAFE_SEGMENT.sub("_", str(raw or ""))[:64]
    return cleaned or fallback


def _enabled() -> bool:
    return bool(getattr(settings, "CHAT_WORKSPACE_ENABLED", True))


def _backend() -> str:
    return str(getattr(settings, "CHAT_WORKSPACE_BACKEND", "docker") or "docker").strip().lower()


def workspace_root(user_id: int, agent_id: int, session_id: str) -> Path:
    """工作区目录：`{base}/{uid}/{aid}/{sid}/`。三级目录即多租户隔离边界。"""
    base = Path(str(getattr(settings, "CHAT_WORKSPACE_DIR", "data/workspaces") or "data/workspaces"))
    return (
        base
        / _safe_segment(str(int(user_id)), "0")
        / _safe_segment(str(int(agent_id)), "0")
        / _safe_segment(session_id, "default")
    )


def open_workspace(
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    job_id: str,
) -> BaseWorkspace | None:
    """创建工作区；关闭或创建失败时返回 None（调用方据此不注册工具）。

    **失败不抛异常**：工作区不可用时对话应当继续（只是没有文件能力），
    而不是让整轮生成失败。只有显式关闭后端时才会走到这里。
    """
    if not _enabled():
        return None
    root = workspace_root(user_id, agent_id, session_id)
    exec_timeout = float(getattr(settings, "CHAT_WORKSPACE_EXEC_TIMEOUT_SECONDS", 30.0) or 30.0)
    read_limit = int(getattr(settings, "CHAT_WORKSPACE_MAX_READ_BYTES", 200_000) or 200_000)

    backend = _backend()
    if backend == "docker":
        try:
            from app.workspace.docker import DockerWorkspace, docker_available, reap_stale_containers

            if docker_available():
                # 清扫上次进程被强杀留下的容器；必须在创建新容器之前
                reaped = reap_stale_containers()
                if reaped:
                    logger.warning("清理残留工作区容器 {} 个", reaped)
                return DockerWorkspace(
                    root,
                    job_id=job_id,
                    image=str(getattr(settings, "CHAT_WORKSPACE_IMAGE", "python:3.11-slim")),
                    mem_limit=str(getattr(settings, "CHAT_WORKSPACE_MEM_LIMIT", "512m")),
                    cpu_limit=float(getattr(settings, "CHAT_WORKSPACE_CPU_LIMIT", 1.0) or 1.0),
                    network=bool(getattr(settings, "CHAT_WORKSPACE_NETWORK", False)),
                    exec_timeout=exec_timeout,
                    read_limit=read_limit,
                )
            logger.warning("Docker 不可用，工作区降级为本地目录（无隔离）")
        except WorkspaceUnavailable as e:
            logger.warning("容器工作区启动失败，降级为本地目录：{}", e)
        except Exception as e:  # noqa: BLE001 - 任何异常都不该中断对话
            logger.warning("工作区初始化异常，降级为本地目录：{}", e)

    # local 兜底：无隔离，仅适合开发或不装 Docker 的部署
    try:
        return LocalWorkspace(root, exec_timeout=exec_timeout, read_limit=read_limit)
    except Exception as e:  # noqa: BLE001
        logger.error("本地工作区创建失败（本轮无文件能力）：{}", e)
        return None


__all__ = [
    "BaseWorkspace",
    "CommandResult",
    "FileInfo",
    "LocalWorkspace",
    "PathEscapeError",
    "WorkspaceError",
    "WorkspaceUnavailable",
    "make_workspace_tools",
    "open_workspace",
    "workspace_root",
]
