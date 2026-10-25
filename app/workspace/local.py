"""本地目录实现：把工作区限制在磁盘上的一个受控目录内。

适用场景：没有 Docker 的环境（或显式关闭容器时的兜底）、以及全部单测。

与 Docker 后端的差别只有一个：`execute` 跑在宿主进程上，没有隔离。
因此本后端**绝不能用于对外服务**，只应作为开发兜底。
"""

from __future__ import annotations

import locale
import os
import signal
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

# 超时后等管道关闭的二次上限（秒）。正常情况下 taskkill /T 已经把整棵树杀干净，
# 管道立刻 EOF，这个 timeout 根本用不到；只在「进程树杀不干净」时才生效，
# 用来保证命令一定能在有限时间内返回，而不是被孙进程拖住。
_KILL_GRACE_SECONDS = 2.0


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """杀掉整棵进程树，而不只是 shell。

    为什么必须杀树：`shell=True` 启动的是 `cmd.exe /c <command>`（POSIX 下是
    `/bin/sh -c`）。`proc.kill()` 只杀掉 shell 本身，它派生出的孙进程继续持有
    stdout/stderr 管道；而 `communicate()` 会一直读到管道 EOF，于是
    「设了 0.5s 超时、实际等了 5.8s」——超时形同虚设（2026-10-24 实测）。

    Windows 用 `taskkill /F /T`（/T 连子孙），POSIX 用独立进程组 + `killpg`。
    两条路径都失败时退回 `proc.kill()`，至少保证 shell 死掉。
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return
        except Exception:  # noqa: BLE001 - 杀不掉就退回单进程 kill，不能让清理本身抛错
            pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except Exception:  # noqa: BLE001 - 同上
            pass
    try:
        proc.kill()
    except Exception:  # noqa: BLE001 - 进程可能已退出
        pass


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
            # 不用 subprocess.run(timeout=)：它在超时后只 kill shell，
            # 孙进程继续持有管道，communicate 会等到孙进程自然结束（实测等满 5.8s）。
            # 这里自己管理生命周期，超时先杀整棵树再收尾。
            proc = subprocess.Popen(  # noqa: S602 - 本来就是执行 shell，由 cwd 锁定在工作区内
                command,
                shell=True,
                cwd=str(work_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=(os.name != "nt"),  # POSIX 需要独立进程组才能 killpg
            )
        except FileNotFoundError as e:
            raise WorkspaceError(f"无法执行命令：{e}") from e

        try:
            raw_out, raw_err = proc.communicate(timeout=limit)
        except subprocess.TimeoutExpired as e:
            # 超时是正常结果而非异常：把已产出的部分还回去，模型才知道跑到哪一步了
            _kill_process_tree(proc)
            try:
                raw_out, raw_err = proc.communicate(timeout=_KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                # 极端情况：树没杀干净（如权限不足）。放弃继续读，关管道避免 fd 泄漏，
                # 用第一次超时时捕获到的部分输出。
                for pipe in (proc.stdout, proc.stderr):
                    try:
                        if pipe is not None:
                            pipe.close()
                    except Exception:  # noqa: BLE001 - 关管道失败不影响结果
                        pass
                raw_out = getattr(e, "output", None) or b""
                raw_err = getattr(e, "stderr", None) or b""
            out, truncated_out = truncate_text(_decode(raw_out or b""), self._output_limit)
            err, truncated_err = truncate_text(_decode(raw_err or b""), self._output_limit)
            return CommandResult(
                stdout=out,
                stderr=(err + f"\n命令执行超时（{limit:g}s），已终止").strip(),
                exit_code=-1,
                duration_ms=int((time.perf_counter() - started) * 1000),
                truncated=truncated_out or truncated_err,
            )

        out, truncated_out = truncate_text(_decode(raw_out), self._output_limit)
        err, truncated_err = truncate_text(_decode(raw_err), self._output_limit)
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
