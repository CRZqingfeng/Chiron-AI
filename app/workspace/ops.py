"""工作区文件操作原语：精确编辑（后续还有内容检索、路径匹配）。

**为什么单独一个模块，而不是继续往 `BaseWorkspace` 上堆方法**

`BaseWorkspace` 是**存储抽象**（读/写/列目录/执行），这里放的是**操作策略**
（按行匹配、跳过规则、裁剪窗口）。两者混在一起会让抽象方法从 5 个涨到 8 个，
将来加新后端（S3 / 远端主机）必须把策略也实现一遍。

本模块只吃一个 `root: Path`，因此 local 与 docker 后端共用——后者继承前者，
且文件读写本来就发生在宿主侧的挂载目录（容器只负责 `execute`），
所以「在容器里跑 grep」这件事既没必要也做不到。

**路径安全**：所有入口先过 `safe_resolve`，它做 `resolve()` + `is_relative_to()`
双重校验，绝对路径、`..` 穿越、指向外部的软链都会被拦下。
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from app.workspace.base import WorkspaceError, safe_resolve

# 单文件编辑上限。为什么不放大：`edit_file` 的语义是「改几行」，
# 真要处理超大文件应该用脚本（run_command）而不是把全文塞进模型上下文。
MAX_EDIT_BYTES = 2 * 1024 * 1024

# 二进制探测窗口：只看前 8KB 是否含 NUL。文本文件里出现 NUL 基本只有二进制一种解释。
_BINARY_PROBE_BYTES = 8192

# 成功时返回的上下文行数（改动前后各几行）
_CONTEXT_LINES = 3


class EditOutcome(TypedDict, total=False):
    """`edit_text` 的结构化结果。

    用 `ok` + `code` 而不是抛异常：错误文案是**给模型看的**，
    属于工具层职责；ops 层只负责判定「是什么情况」，不关心怎么措辞。
    这样同一套判定可以被别的调用方（比如将来的前端预览）复用。
    """

    ok: bool
    # 失败原因代码：no_file / is_dir / too_large / not_text / empty_old /
    # same / unsafe_replace_all / not_found / ambiguous
    code: str
    occurrences: int
    first_line: int
    total_lines: int
    # "CRLF" / "LF"，供工具层在返回里告知模型
    newline: str
    # 多处匹配时的全部行号（仅 ambiguous 用得到）
    match_lines: list[int]
    before: list[tuple[int, str]]
    after: list[tuple[int, str]]
    # 非致命提示，例如「文件原本混用换行，已统一为 CRLF」
    note: str


def detect_newline(text: str) -> tuple[str, bool]:
    """返回 (主导行尾符, 是否混用)。

    只判主导，不做逐行还原：真实项目里混用换行的文件极少，
    为它把匹配逻辑复杂化不值得；但**必须如实报告**，否则我们会在用户不知情时
    悄悄改动他没碰过的行。
    """
    crlf = text.count("\r\n")
    lf_only = text.count("\n") - crlf
    if crlf and lf_only:
        return ("\r\n" if crlf >= lf_only else "\n"), True
    if crlf:
        return "\r\n", False
    return "\n", False


def _normalize_newlines(text: str) -> str:
    """统一成 `\\n` 以便匹配。

    这是「模型给的 old_string 在 CRLF 文件上匹配不上」这个高频失败模式的解法：
    模型从 `read_file` 看到的是一行行文本，它复制回来的片段行尾是 `\\n`，
    而磁盘上是 `\\r\\n`。字符级比较必然失败，报错却是「未找到」——
    模型会以为文件被改了，然后开始瞎试。
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def is_binary(raw: bytes) -> bool:
    """按前 8KB 是否含 NUL 判定二进制。"""
    return b"\x00" in raw[:_BINARY_PROBE_BYTES]


def _line_of(text: str, pos: int) -> int:
    """字符偏移 → 1-based 行号。"""
    return text.count("\n", 0, pos) + 1


def _context_before(lines: list[str], line_no: int, count: int = _CONTEXT_LINES) -> list[tuple[int, str]]:
    start = max(1, line_no - count)
    return [(n, lines[n - 1]) for n in range(start, line_no)]


def _context_after(lines: list[str], line_no: int, count: int = _CONTEXT_LINES) -> list[tuple[int, str]]:
    end = min(len(lines), line_no + count - 1)
    return [(n, lines[n - 1]) for n in range(line_no, end + 1)]


def edit_text(
    root: Path,
    path: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
) -> EditOutcome:
    """在 `path` 里把 `old_string` 精确替换成 `new_string`。

    匹配在**行尾归一化后的文本**上做（见 `_normalize_newlines`），
    因此既能替换行内片段，也能跨行替换，且不受 CRLF/LF 差异影响。
    写回时统一用文件原有的主导行尾符。
    """
    try:
        target = safe_resolve(root, path)
    except WorkspaceError as e:
        return EditOutcome(ok=False, code="bad_path", note=str(e))

    if not target.exists():
        return EditOutcome(ok=False, code="no_file")
    if target.is_dir():
        return EditOutcome(ok=False, code="is_dir")

    size = target.stat().st_size
    if size > MAX_EDIT_BYTES:
        return EditOutcome(ok=False, code="too_large", note=str(size))

    raw = target.read_bytes()
    if is_binary(raw):
        return EditOutcome(ok=False, code="not_text")

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        # 严格解码：用 replace 兜底会把原文写坏（U+FFFD 落盘就回不去了）
        return EditOutcome(ok=False, code="not_text")

    old_n = _normalize_newlines(old_string or "")
    new_n = _normalize_newlines(new_string or "")

    if not old_n:
        return EditOutcome(ok=False, code="empty_old")
    if old_n == new_n:
        return EditOutcome(ok=False, code="same")
    if replace_all and not old_n.strip():
        # 纯空白 + 全量替换会吃掉大量无关空行，几乎不可能是本意
        return EditOutcome(ok=False, code="unsafe_replace_all")

    normalized = _normalize_newlines(content)
    newline, mixed = detect_newline(content)
    # 用 splitlines 而不是 count("\n")+1：末尾有换行的文件会多算一个空行，
    # 报给模型的行数应当与它 read_file 看到的一致
    total_lines = max(1, len(normalized.splitlines()))

    positions: list[int] = []
    start = 0
    while True:
        hit = normalized.find(old_n, start)
        if hit < 0:
            break
        positions.append(hit)
        # 从上一次命中之后再找，避免重叠导致的重复计数
        start = hit + len(old_n)

    if not positions:
        return EditOutcome(ok=False, code="not_found", total_lines=total_lines)

    match_lines = [_line_of(normalized, p) for p in positions]
    if len(positions) > 1 and not replace_all:
        return EditOutcome(
            ok=False,
            code="ambiguous",
            occurrences=len(positions),
            match_lines=match_lines,
            total_lines=total_lines,
        )

    first_pos = positions[0]
    first_line = match_lines[0]

    if replace_all:
        result_n = normalized.replace(old_n, new_n)
    else:
        result_n = normalized[:first_pos] + new_n + normalized[first_pos + len(old_n):]

    # 写回：把归一化后的 \n 还原成文件原有行尾符
    result = result_n if newline == "\n" else result_n.replace("\n", newline)
    try:
        target.write_bytes(result.encode("utf-8"))
    except OSError as e:
        # 磁盘满 / 只读挂载 / 权限：都不能让整轮生成失败，如实告知即可
        return EditOutcome(ok=False, code="write_failed", note=str(e))

    src_lines = normalized.splitlines()
    out_lines = result_n.splitlines()

    newline_label = "CRLF" if newline != "\n" else "LF"
    note = f"文件原本混用换行，已统一为 {newline_label}" if mixed else ""

    return EditOutcome(
        ok=True,
        occurrences=len(positions),
        first_line=first_line,
        total_lines=total_lines,
        newline=newline_label,
        before=_context_before(src_lines, first_line),
        after=_context_after(out_lines, first_line),
        note=note,
    )


__all__ = [
    "MAX_EDIT_BYTES",
    "EditOutcome",
    "detect_newline",
    "edit_text",
    "is_binary",
]
