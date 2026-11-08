"""工作区产出物的只读出口（U10 P0-C）。

**为什么单独一个 router，而不是塞进 chat.py**：chat.py 已经 700+ 行，
再放三个文件端点会让它更难读；而且这三个端点与对话链路无关，
只依赖工作区目录，单独成文件更容易被审。

**为什么不启动容器**：工作区文件本来就在宿主侧——`DockerWorkspace` 继承
`LocalWorkspace`，只重写了 `execute`（容器只负责跑命令）。所以列目录/读文件
用一个只读的 `LocalWorkspace` 实例就够。为一次列表去起一个 Docker 容器，
既慢又没必要。

**安全边界（三条，缺一不可）**

1. 三级目录由服务端用 token 里的 `user_id` + 校验过的 `agent_id` + `session_id` 拼，
   **绝不接受前端传 uid**——否则改一个 query 参数就能读别人的工作区。
2. 路径一律走 `safe_resolve`，越界返回 **403** 而不是 404：
   404 会把「存在但你没权限」与「不存在」区分开，等于泄露存在性。
3. **只读**。不提供写/删/重命名接口。
"""

from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.dependency import DependAuth
from app.models.admin import User
from app.schemas.base import Fail, Success
from app.services.user_agent_access import get_accessible_agent
from app.workspace import workspace_root
from app.workspace.base import PathEscapeError, WorkspaceError, safe_resolve
from app.workspace.local import LocalWorkspace

router = APIRouter()

# 在线预览上限：再大多半是数据文件或日志，直接下载更合适
MAX_PREVIEW_BYTES = 1024 * 1024
# 单文件下载上限：与 P0 方案一致
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
# 列目录单层上限
MAX_LIST_ITEMS = 500

# 在线预览白名单后缀。不在表里的一律引导下载——
# 用 mimetypes 猜「是不是文本」不够可靠（.bin/.dat 也会被猜成 octet-stream，
# 而真正需要预览的就是这几类）
_PREVIEW_SUFFIXES = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".vue", ".html", ".htm",
    ".css", ".scss", ".less", ".sh", ".bash", ".bat", ".ps1", ".sql",
    ".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".rb", ".php", ".kt",
    ".xml", ".svg", ".gitignore", ".dockerignore",
})


def _open_readonly(user_id: int, agent_id: int, session_id: str) -> LocalWorkspace | None:
    """以只读方式打开会话工作区；目录不存在返回 None。

    不存在的目录**不创建**：`LocalWorkspace.__init__` 会 `mkdir`，
    而「用户只是点开文件抽屉看一眼」不该在磁盘上留下空目录。
    """
    root = workspace_root(user_id, agent_id, session_id)
    if not root.exists() or not root.is_dir():
        return None
    return LocalWorkspace(root)


def _resolve(ws: LocalWorkspace, path: str) -> Path:
    """把用户给的相对路径解析成绝对路径；越界抛 403。

    这里刻意用 `safe_resolve` 而不是 `ws.list_dir`——后者内部也会校验，
    但校验失败的异常类型与「目录不存在」一样，分不出来该返 403 还是 404。
    """
    try:
        return safe_resolve(ws.root_path, path or ".")
    except PathEscapeError as e:
        # `e` 的文案已经是「路径超出工作区范围：xxx」，不要再包一层前缀
        raise HTTPException(status_code=403, detail=str(e)) from e


@router.get("/chat/workspace/files", summary="列出会话工作区文件", tags=["智能体模块"])
async def list_workspace_files(
    agent_id: int = Query(..., description="智能体 ID"),
    session_id: str = Query(..., description="会话 ID"),
    path: str = Query("", description="相对工作区根目录的子目录，留空为根目录"),
    current_user: User = DependAuth,
):
    """列出某个目录的直接子项（单层，不递归）。

    工作区目录不存在时返回空列表而不是 404：用户还没让 agent 产出任何文件，
    这是正常状态，不是错误。
    """
    user_id = int(current_user.id)
    ua = await get_accessible_agent(agent_id, user_id)
    if not ua:
        return Fail(code=404, msg="智能体不存在或无权限访问")

    ws = await asyncio.to_thread(_open_readonly, user_id, agent_id, session_id)
    if ws is None:
        return Success(data={"path": path or "", "items": [], "truncated": False})

    target = _resolve(ws, path)
    if not target.exists():
        return Fail(code=404, msg="目录不存在")
    if not target.is_dir():
        return Fail(code=400, msg="path 需要是目录")

    try:
        items = await asyncio.to_thread(ws.list_dir, path or ".")
    except WorkspaceError as e:
        return Fail(code=400, msg=str(e))

    # 隐藏项不展示：`.env` 之类可能含密钥，点开就等于泄露给浏览器历史
    # 补上 `name`：`list_dir` 只给相对 `path`，前端列表要单独的名字
    visible = [
        {**it, "name": Path(it["path"]).name}
        for it in items
        if not Path(it["path"]).name.startswith(".")
    ]
    truncated = len(visible) > MAX_LIST_ITEMS
    visible = visible[:MAX_LIST_ITEMS]

    return Success(
        data={
            "path": path or "",
            "items": visible,
            "truncated": truncated,
            "total": len(visible),
        }
    )


@router.get("/chat/workspace/file", summary="预览会话工作区文本文件", tags=["智能体模块"])
async def preview_workspace_file(
    agent_id: int = Query(..., description="智能体 ID"),
    session_id: str = Query(..., description="会话 ID"),
    path: str = Query(..., description="相对工作区根目录的文件路径"),
    current_user: User = DependAuth,
):
    """返回文本内容。超 1MB 只返回前 1MB，并用 `truncated` 标记。"""
    user_id = int(current_user.id)
    ua = await get_accessible_agent(agent_id, user_id)
    if not ua:
        return Fail(code=404, msg="智能体不存在或无权限访问")

    ws = await asyncio.to_thread(_open_readonly, user_id, agent_id, session_id)
    if ws is None:
        return Fail(code=404, msg="工作区还没有任何文件")

    target = _resolve(ws, path)
    if not target.exists() or not target.is_file():
        return Fail(code=404, msg="文件不存在")

    suffix = target.suffix.lower()
    if suffix not in _PREVIEW_SUFFIXES:
        return Fail(code=400, msg=f"不支持在线预览 {suffix or '该类型'} 文件，请用下载")

    size = target.stat().st_size
    raw = await asyncio.to_thread(target.read_bytes)
    truncated = size > MAX_PREVIEW_BYTES
    if truncated:
        raw = raw[:MAX_PREVIEW_BYTES]

    return Success(
        data={
            "path": path,
            "content": raw.decode("utf-8", errors="replace"),
            "size": size,
            "truncated": truncated,
        }
    )


@router.get("/chat/workspace/download", summary="下载会话工作区文件", tags=["智能体模块"])
async def download_workspace_file(
    agent_id: int = Query(..., description="智能体 ID"),
    session_id: str = Query(..., description="会话 ID"),
    path: str = Query(..., description="相对工作区根目录的文件路径"),
    current_user: User = DependAuth,
):
    """以附件形式返回文件本体。

    中文文件名交给 `FileResponse(filename=...)` 处理——Starlette 会同时输出
    `filename` 与 RFC 5987 的 `filename*=utf-8''...`，不要手写这个头，
    手写版本在旧浏览器上会把中文变成乱码。
    """
    user_id = int(current_user.id)
    ua = await get_accessible_agent(agent_id, user_id)
    if not ua:
        raise HTTPException(status_code=404, detail="智能体不存在或无权限访问")

    ws = await asyncio.to_thread(_open_readonly, user_id, agent_id, session_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="工作区还没有任何文件")

    target = _resolve(ws, path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")

    size = target.stat().st_size
    if size > MAX_DOWNLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{size / 1024 / 1024:.1f}MB），单文件下载上限 20MB",
        )

    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(
        path=str(target),
        media_type=media_type,
        filename=target.name,
        headers={"Cache-Control": "private, no-store"},
    )
