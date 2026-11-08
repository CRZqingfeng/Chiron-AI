"""会话附件 → 会话工作区的镜像（U10 P1-D）。

**要解决的问题**：附件存在对象存储（MinIO），工作区是另一个挂载目录，两者原本
没有任何桥接。用户传一个 5MB CSV 说「用 python 分析它」——容器里根本没有这个文件。
唯一的变通路径是 `read_session_attachment`（单次截断 16k）→ 模型复述 → `write_file`，
大文件必断，而且复述本身就必然失真。

**做法**：上传成功后在会话工作区写一份 `_attachments/<原文件名>`，会话删除时一并清理。
同名文件覆盖（用户重传同一份时，最新版本生效）。

**三条边界**
1. 落盘失败**绝不抛异常**：它是增强能力，不能因为它让上传失败
2. 单会话镜像总配额 200MB，超了跳过并记日志（上传本身照常成功）
3. 文件名保留中文，只剔除路径分隔符与控制字符——附件名是给人看的
"""

from __future__ import annotations

import re
from pathlib import Path

from app.log import logger
from app.settings import settings
from app.workspace import workspace_root
from app.workspace.base import safe_resolve

# 镜像目录名。用 `_` 开头而不是 `.`：`.` 开头的目录在文件抽屉里会被隐藏，
# 而用户应该能看到自己传上来的文件
ATTACHMENT_DIR = "_attachments"

# 单会话镜像总配额
MAX_MIRROR_BYTES = 200 * 1024 * 1024

# 只剔除真正危险/非法的字符；中文、空格、括号都保留
_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def _safe_name(filename: str) -> str:
    """取纯文件名并剔除危险字符，保留中文。"""
    base = Path(str(filename or "file")).name
    cleaned = _UNSAFE_CHARS.sub("_", base).strip(". ") or "file"
    return cleaned[:120]


def _dir_size(path: Path) -> int:
    total = 0
    try:
        entries = list(path.rglob("*"))
    except OSError:
        return 0
    for item in entries:
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _enabled() -> bool:
    return bool(getattr(settings, "CHAT_WORKSPACE_ENABLED", True))


def sync_attachment_to_workspace(
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    filename: str,
    raw: bytes,
) -> str | None:
    """把附件写进会话工作区；返回相对路径，跳过或失败时返回 None。"""
    if not _enabled() or not raw:
        return None

    try:
        root = workspace_root(int(user_id), int(agent_id), session_id)
        name = _safe_name(filename)
        target = safe_resolve(root, f"{ATTACHMENT_DIR}/{name}")

        mirror_dir = target.parent
        existing = target.stat().st_size if target.exists() else 0
        # 覆盖同名文件时，原文件占用的空间会让出来，所以先减掉
        used = _dir_size(mirror_dir) - existing if mirror_dir.exists() else 0
        if used + len(raw) > MAX_MIRROR_BYTES:
            logger.warning(
                "附件镜像超出会话配额，跳过落盘：session={} file={} 已用={}MB",
                session_id,
                name,
                used // (1024 * 1024),
            )
            return None

        mirror_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return f"{ATTACHMENT_DIR}/{target.name}"
    except Exception as e:  # noqa: BLE001 - 落盘是增强能力，任何异常都不能影响上传
        logger.warning("附件写入工作区失败（已忽略）：session={} file={}", session_id, filename)
        logger.debug("附件落盘异常详情：{}", e)
        return None


def _purge_mirror_dir(mirror_dir: Path) -> None:
    """删掉一个镜像目录及其内容；目录不存在或删不干净都不报错。"""
    if not mirror_dir.exists():
        return
    for item in mirror_dir.rglob("*"):
        if item.is_file():
            item.unlink(missing_ok=True)
    # 由深到浅删空目录
    for sub in sorted(mirror_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if sub.is_dir():
            try:
                sub.rmdir()
            except OSError:
                pass
    try:
        mirror_dir.rmdir()
    except OSError:
        pass


def purge_workspace_attachments(user_id: int, agent_id: int, session_id: str) -> None:
    """会话删除时清掉镜像目录。失败只记日志——清理不该阻断删除主流程。"""
    try:
        root = workspace_root(int(user_id), int(agent_id), session_id)
        _purge_mirror_dir(safe_resolve(root, ATTACHMENT_DIR))
    except Exception as e:  # noqa: BLE001
        logger.warning("清理工作区附件镜像失败（已忽略）：session={}", session_id)
        logger.debug("清理异常详情：{}", e)


def purge_workspace_attachments_for_agent(user_id: int, agent_id: int) -> None:
    """智能体删除时清掉它名下所有会话的镜像目录。"""
    try:
        # `workspace_root` 的第三段就是会话目录，取 parent 即 {base}/{uid}/{aid}
        agent_dir = workspace_root(int(user_id), int(agent_id), "_").parent
        if not agent_dir.exists():
            return
        for session_dir in agent_dir.iterdir():
            if session_dir.is_dir():
                _purge_mirror_dir(session_dir / ATTACHMENT_DIR)
    except Exception as e:  # noqa: BLE001
        logger.warning("清理智能体工作区附件镜像失败（已忽略）：agent={}", agent_id)
        logger.debug("清理异常详情：{}", e)


def attachment_mirror_names(user_id: int, agent_id: int, session_id: str) -> list[str]:
    """镜像目录里现有的文件名；目录不存在时返回空列表。"""
    try:
        root = workspace_root(int(user_id), int(agent_id), session_id)
        mirror_dir = safe_resolve(root, ATTACHMENT_DIR)
        if not mirror_dir.exists():
            return []
        return sorted(p.name for p in mirror_dir.iterdir() if p.is_file())
    except Exception:  # noqa: BLE001
        return []


def attachment_mirror_hint(user_id: int, agent_id: int, session_id: str) -> str:
    """只在镜像里**真有文件**时才告诉模型「可以直接读」。

    写死这句话会在两种情况下变成假话：工作区没启用、或落盘因配额被跳过。
    注入提示必须由内容决定，不能由「我们本来打算做这件事」决定。
    """
    if not _enabled():
        return ""
    names = attachment_mirror_names(user_id, agent_id, session_id)
    if not names:
        return ""
    shown = "、".join(names[:8])
    more = f" 等 {len(names)} 个" if len(names) > 8 else ""
    return (
        f"以上附件已同步到工作区 `{ATTACHMENT_DIR}/` 目录（{shown}{more}），"
        f"可以直接用 workspace_run_command 跑脚本读取，例如 "
        f"`python -c \"open('{ATTACHMENT_DIR}/{names[0]}')\"`；"
        "处理大文件（CSV/日志）走这条路径，不要用 read_session_attachment 分段读。"
    )


__all__ = [
    "ATTACHMENT_DIR",
    "MAX_MIRROR_BYTES",
    "attachment_mirror_hint",
    "attachment_mirror_names",
    "purge_workspace_attachments",
    "purge_workspace_attachments_for_agent",
    "sync_attachment_to_workspace",
]
