"""本地目录实现：把工作区限制在磁盘上的一个受控目录内。

适用场景：没有 Docker 的环境（或显式关闭容器时的兜底）、以及全部单测。

与 Docker 后端的差别只有一个：`execute` 跑在宿主进程上，没有隔离。
因此本后端**绝不能用于对外服务**，只应作为开发兜底。
"""

from __future__ import annotations

import locale
import subprocess
import time
from pathlib import Path

from app.workspace.base import (
    BaseWorkspace,
    CommandResult,
    FileInfo,
    PathEscapeError,
    WorkspaceError,
    safe_resolve,
    truncate_text,
)


def _decode(raw: bytes) -> str:
    """解码子进程输出。

    顺序有讲究：先 UTF-8（Linux 与多数现代 CLI），再系统默认编码
    （中文 Windows 的 cmd 输出常是 GBK），最后用 replace 兜底——
    绝不能让一个乱码字节让整次工具调用失败。
    """
    if not raw:
        return ""
    for enc in ("utf-8", locale.getpreferredencoding(False), "gbk"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


class LocalWorkspace(BaseWorkspace):
    """在宿主目录中执行。`root` 会被创建（含父目录）。"""

    def __init__(
        self,
        root: Path | str,
        *,
        exec_timeout: float = 30.0,
        output_limit: int = 32_000,
        read_limit: int = 200_000,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._exec_timeout = float(exec_timeout)
        self._output_limit = int(output_limit)
        self._read_limit = int(read_limit)

    # --- 内部 -------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        return safe_resolve(self.root, path)

    def _rel(self, p: Path) -> str:
        """转成相对路径 + POSIX 分隔符，避免 Windows 反斜杠泄漏给模型。"""
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            return p.as_posix()

    # --- 接口实现 ---------------------------------------------------------

    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        work_dir = self._resolve(cwd) if cwd else self.root
        if not work_dir.exists() or not work_dir.is_dir():
            raise WorkspaceError(f"工作目录不存在：{cwd}")
        limit = float(timeout if timeout is not None else self._exec_timeout)
        started = time.perf_counter()
        try:
            proc = subprocess.run(  # noqa: S602 - 本来就是执行 shell，由 cwd 锁定在工作区内
                command,
                shell=True,
                cwd=str(work_dir),
                capture_output=True,
                timeout=limit,
            )
        except subprocess.TimeoutExpired as e:
            # 超时是正常结果而非异常：把已产出的部分还回去，模型才知道跑到哪一步了
            out, truncated_out = truncate_text(_decode(e.stdout or b""), self._output_limit)
            err, truncated_err = truncate_text(_decode(e.stderr or b""), self._output_limit)
            return CommandResult(
                stdout=out,
                stderr=(err + f"\n命令执行超时（{limit:.0f}s），已终止").strip(),
                exit_code=-1,
                duration_ms=int((time.perf_counter() - started) * 1000),
                truncated=truncated_out or truncated_err,
            )
        except FileNotFoundError as e:
            raise WorkspaceError(f"无法执行命令：{e}") from e

        out, truncated_out = truncate_text(_decode(proc.stdout), self._output_limit)
        err, truncated_err = truncate_text(_decode(proc.stderr), self._output_limit)
        return CommandResult(
            stdout=out,
            stderr=err,
            exit_code=int(proc.returncode),
            duration_ms=int((time.perf_counter() - started) * 1000),
            truncated=truncated_out or truncated_err,
        )

    def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise WorkspaceError(f"文件不存在：{path}")
        if target.is_dir():
            raise WorkspaceError(f"{path} 是目录，请用 list_dir")
        limit = int(max_bytes if max_bytes is not None else self._read_limit)
        # 先按字节读上限，避免把上百 MB 的文件整个读进内存
        with open(target, "rb") as fh:
            raw = fh.read(limit + 1 if limit > 0 else -1)
        clipped = len(raw) > limit > 0
        text = _decode(raw[:limit] if clipped else raw)
        if clipped:
            text += f"\n...[文件较大，仅显示前 {limit} 字节]"
        return text

    def write_text(self, path: str, content: str) -> int:
        target = self._resolve(path)
        if target.is_dir():
            raise WorkspaceError(f"{path} 是目录，无法写入")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (content or "").encode("utf-8")
        target.write_bytes(data)
        return len(data)

    def list_dir(self, path: str = ".") -> list[FileInfo]:
        target = self._resolve(path)
        if not target.exists():
            raise WorkspaceError(f"目录不存在：{path}")
        if not target.is_dir():
            raise WorkspaceError(f"{path} 不是目录，请用 read_file")
        items: list[FileInfo] = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            try:
                stat = child.stat()
            except OSError:
                # 权限或竞态导致取不到：跳过而不是整次调用失败
                continue
            items.append(
                FileInfo(
                    path=self._rel(child),
                    is_dir=child.is_dir(),
                    size=int(stat.st_size),
                    mtime=float(stat.st_mtime),
                )
            )
        return items

    def cleanup(self) -> None:
        """本地后端无资源需要释放；工作区文件保留（跨轮持久）。"""
        return None


__all__ = ["LocalWorkspace", "PathEscapeError"]
