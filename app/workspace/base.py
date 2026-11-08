"""工作区抽象：让 agent 能在受控环境里读写文件、执行命令。

为什么需要它：改造前 agent 只能「说」不能「动手」——九类工具全是检索与联网，
没有任何一个能在磁盘上产生副作用。工作区层补的就是这一块。

设计取舍（容易走错的地方）：

1. **接口是内容级而不是路径级**。OpenHands 用 `file_upload(source_path, dest_path)`，
   那是为「远程容器 + HTTP 传输」设计的，文件必须经本地路径中转。我们的调用方是
   LLM 工具，输入输出都是文本，中转本地文件纯属浪费；`local` 后端下内容级接口就是
   一次 `Path.read_text()`，零进程开销。

2. **`execute` 在 U4 只实现不暴露**。接口两个后端都实现且有测试，但不注册成工具——
   否则在 U5b（审批中断）落地之前，模型可以执行任意 shell，中间存在无保护窗口。

3. **路径校验必须在实现内部**，不能依赖调用方自觉。所有文件方法都先过
   `safe_resolve`，它做 `resolve()` + `is_relative_to()` 双重校验，符号链接一并解决
   （软链指向外部时 resolve 后的真实路径在根外，直接拒绝）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict


class WorkspaceError(RuntimeError):
    """工作区操作失败的基类。"""


class PathEscapeError(WorkspaceError):
    """请求的路径解析后落在工作区根目录之外（含目录穿越与符号链接逃逸）。"""


class WorkspaceUnavailable(WorkspaceError):
    """工作区无法创建或不可用（例如 Docker 后端启动容器失败）。"""


class CommandResult(TypedDict):
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    # 输出是否被截断。为 True 时调用方应意识到「看到的不是全部」，
    # 否则模型会基于半截日志得出错误结论。
    truncated: bool


class FileInfo(TypedDict):
    # 相对工作区根目录的路径，统一用 POSIX 分隔符，避免 Windows 反斜杠泄漏给模型
    path: str
    is_dir: bool
    size: int
    mtime: float


def safe_resolve(root: Path, user_path: str) -> Path:
    """把用户给的相对路径解析成根目录下的绝对路径，越界即抛 `PathEscapeError`。

    三类攻击都要挡住：
    - `../x`、`a/../../x` 这类相对穿越
    - `/etc/passwd` 这类绝对路径（`root / "/etc/passwd"` 在 pathlib 下会直接变成
      `/etc/passwd`，是 Python 里最容易被忽略的坑）
    - 工作区内一个指向外部的符号链接（`resolve()` 解链后落在根外）

    注意 `resolve()` 对不存在的路径不会抛错（strict=False 默认值），
    所以 `write_file` 创建新文件时也能正常校验。
    """
    raw = (user_path or "").strip()
    if not raw:
        raise PathEscapeError("路径不能为空")
    # 先挡绝对路径：Windows 盘符与 POSIX 绝对路径都覆盖
    candidate = Path(raw)
    if candidate.is_absolute() or candidate.drive:
        raise PathEscapeError(f"禁止使用绝对路径：{raw}")

    root_resolved = root.resolve()
    target = (root_resolved / raw).resolve()
    if not target.is_relative_to(root_resolved):
        raise PathEscapeError(f"路径超出工作区范围：{raw}")
    return target


def truncate_text(text: str, limit: int) -> tuple[str, bool]:
    """按字符数截断并返回 (文本, 是否发生截断)。

    用字符而不是字节做单位：模型按 token 计费，字符数与 token 数的关系更稳定，
    而字节数会被中文（3 字节/字）严重高估。
    """
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit] + f"\n...[输出已截断，共 {len(text)} 字符，仅显示前 {limit} 字符]", True


class BaseWorkspace(ABC):
    """工作区接口。两个后端（local / docker）实现同一套语义。

    所有文件路径参数都是**相对工作区根目录**的相对路径；绝对路径一律拒绝。
    """

    @abstractmethod
    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        """在 `cwd`（默认根目录）执行 shell 命令。

        实现必须：超时后终止进程、截断超大输出、不因命令返回非 0 而抛异常
        （非 0 是正常结果，调用方要看 `exit_code`）。
        """

    @abstractmethod
    def read_text(self, path: str, *, max_bytes: int = 200_000) -> str:
        """读取文本文件，超过 `max_bytes` 时截断并给出提示。"""

    @abstractmethod
    def write_text(self, path: str, content: str) -> int:
        """写入文本（覆盖），父目录自动创建，返回写入的字节数。"""

    @abstractmethod
    def list_dir(self, path: str = ".") -> list[FileInfo]:
        """列出目录的直接子项；路径不存在或不是目录时抛 `WorkspaceError`。"""

    @abstractmethod
    def cleanup(self) -> None:
        """释放资源（停容器）。必须幂等，且不删除工作区文件——跨轮持久是本层目标之一。"""

    @property
    @abstractmethod
    def root_path(self) -> Path:
        """工作区根目录的绝对路径。

        给 `app/workspace/ops.py` 的纯函数用（精确编辑 / 内容检索 / 路径匹配）。
        那些操作需要自己遍历目录树、按行切分原文，而现有接口都不合适：
        `list_dir` 只列一层，`read_text` 会追加截断提示从而污染内容。

        暴露路径而不是把操作扁平化成抽象方法，是为了让「存储」与「策略」分开：
        新增后端只需提供一个根目录，不必把匹配规则重写一遍。
        """
