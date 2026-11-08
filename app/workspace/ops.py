"""工作区文件操作原语：精确编辑、内容检索、路径匹配。

**为什么单独一个模块，而不是继续往 `BaseWorkspace` 上堆方法**

`BaseWorkspace` 是**存储抽象**（读/写/列目录/执行），这里放的是**操作策略**
（匹配规则、跳过规则、裁剪窗口）。两者混在一起会让抽象方法从 5 个涨到 8 个，
将来加新后端（S3 / 远端主机）必须把策略也实现一遍。

本模块只吃一个 `root: Path`，因此 local 与 docker 后端共用——后者继承前者，
且文件读写本来就发生在宿主侧的挂载目录（容器只负责 `execute`），
所以「在容器里跑 grep」这件事既没必要也做不到。

**路径安全**：所有入口先过 `safe_resolve`，它做 `resolve()` + `is_relative_to()`
双重校验，绝对路径、`..` 穿越、指向外部的软链都会被拦下。
"""

from __future__ import annotations

import fnmatch
import os
import re
import time
from pathlib import Path
from typing import Iterator, TypedDict

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


def _find_all(haystack: str, needle: str) -> list[int]:
    """收集所有不重叠的出现位置。"""
    positions: list[int] = []
    if not needle:
        return positions
    start = 0
    while True:
        hit = haystack.find(needle, start)
        if hit < 0:
            break
        positions.append(hit)
        start = hit + len(needle)
    return positions


# 匹配 read_file 输出里的行号栏，形如 `    12 | content`
_LINE_PREFIX_RE = re.compile(r"^\s*\d+\s*\|\s?")


def strip_line_number_prefix(text: str) -> str | None:
    """把模型误带上的行号前缀剥掉；不像行号栏就返回 None。

    为什么需要：`read_file` 的输出带行号（ACI 要求模型能引用行），但模型复制片段
    去做 `edit_file` 时很容易把 `   87 | ` 一起带上——那是文件里根本不存在的字符，
    匹配必然失败，而报错只会说「未找到」，模型不知道毛病出在自己多带了前缀。

    只在**每一行**都符合 `数字 | ` 形状时才认为它是行号栏：表格类内容
    （`  12 | 值`）也可能长成这样，所以剥离后仍匹配不上时调用方会回退原样。
    """
    if not text:
        return None
    rows = text.split("\n")
    candidates = [r for r in rows if r.strip()]
    if not candidates or not all(_LINE_PREFIX_RE.match(r) for r in candidates):
        return None
    return "\n".join(_LINE_PREFIX_RE.sub("", r) for r in rows)


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

    positions = _find_all(normalized, old_n)

    if not positions:
        # 兜底：模型可能把 read_file 输出里的行号栏一起复制进来了。
        # 剥一次再试，仍失败就按原样报「未找到」——绝不能因为剥离而误改别处。
        alt_old = strip_line_number_prefix(old_n)
        if alt_old and alt_old != old_n:
            alt_positions = _find_all(normalized, alt_old)
            if alt_positions:
                old_n = alt_old
                new_n = strip_line_number_prefix(new_n) or new_n
                positions = alt_positions

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


# --- 读取：行级分页 -------------------------------------------------------

# 读取上限比编辑宽松：读完不改，5MB 足够覆盖工作区里绝大多数文本产物
MAX_READ_BYTES = 5 * 1024 * 1024
MAX_READ_LINES = 5000
# 行号栏宽度固定，模型对齐的行号更容易定位（宽度随文件变会让输出抖动）
_GUTTER_WIDTH = 6


class ReadWindow(TypedDict, total=False):
    ok: bool
    code: str
    # (行号, 该行内容)，行号 1-based
    lines: list[tuple[int, str]]
    total_lines: int
    start_line: int
    end_line: int
    has_more: bool
    # "chars"：因单次长度上限停下；""：读到窗口末尾
    stop_reason: str
    note: str


def read_window(
    root: Path,
    path: str,
    *,
    offset: int = 1,
    limit: int = 2000,
    max_chars: int = 20_000,
) -> ReadWindow:
    """按**行**读取一个窗口。

    为什么是行而不是字符：模型引用位置、生成编辑指令都靠行号；字符级分页
    既没法给行号，也没法让模型「从第 87 行接着读」。这是 ACI 的基本要求。

    `offset` 用 **1-based**（第 1 行就是文件第一行）：模型看到的是 1-based 行号，
    说「从第 87 行开始」也是 1-based，用 0-based 会让它系统性差一行。
    """
    try:
        target = safe_resolve(root, path)
    except WorkspaceError as e:
        return ReadWindow(ok=False, code="bad_path", note=str(e))

    if not target.exists():
        return ReadWindow(ok=False, code="no_file")
    if target.is_dir():
        return ReadWindow(ok=False, code="is_dir")

    try:
        size = target.stat().st_size
    except OSError:
        return ReadWindow(ok=False, code="no_file")
    if size > MAX_READ_BYTES:
        return ReadWindow(ok=False, code="too_large", note=str(size))

    raw = target.read_bytes()
    if is_binary(raw):
        return ReadWindow(ok=False, code="not_text")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ReadWindow(ok=False, code="not_text")

    all_lines = content.splitlines()
    total = len(all_lines)
    if total == 0:
        return ReadWindow(
            ok=True, lines=[], total_lines=0, start_line=0, end_line=0, has_more=False
        )

    try:
        start = max(1, int(offset))
    except (TypeError, ValueError):
        start = 1
    try:
        span = max(1, min(int(limit), MAX_READ_LINES))
    except (TypeError, ValueError):
        span = 2000
    try:
        budget = max(500, int(max_chars))
    except (TypeError, ValueError):
        budget = 20_000

    if start > total:
        return ReadWindow(ok=False, code="offset_out_of_range", total_lines=total)

    end = min(total, start + span - 1)

    window: list[tuple[int, str]] = []
    used = 0
    cursor = start - 1
    stop_reason = ""
    for n in range(start, end + 1):
        raw_line = all_lines[n - 1]
        cost = _GUTTER_WIDTH + 3 + len(raw_line) + 1

        if not window:
            # 第一行就超预算：单行 CSV / 压缩过的 JSON 会一次吐出几万字符。
            # 这里必须截断这一行——但行内没法续读，所以把 stop_reason 标成
            # long_line，由工具层引导模型改用 grep 定位，而不是给一个假的 offset。
            if cost > budget:
                keep = max(200, budget - _GUTTER_WIDTH - 40)
                window.append(
                    (n, raw_line[:keep] + f" …（本行共 {len(raw_line)} 字符，已截断）")
                )
                cursor = n
                stop_reason = "long_line"
                break
            window.append((n, raw_line))
            used += cost
            cursor = n
            continue

        if used + cost > budget:
            stop_reason = "chars"
            break
        window.append((n, raw_line))
        used += cost
        cursor = n

    return ReadWindow(
        ok=True,
        lines=window,
        total_lines=total,
        start_line=start,
        end_line=cursor,
        has_more=cursor < total,
        stop_reason=stop_reason,
    )


# --- 检索：grep / glob ----------------------------------------------------

# 目录剪枝。用 `os.walk` 而不是 `Path.rglob` 就是为了能剪枝——
# rglob 会把 node_modules 和 .git 一起扫完，深目录能直接扫到超时。
SKIP_DIRS = frozenset({
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    ".next",
})

# 一次调用的扫描上限：模型不该能把 CPU 占满，用户也不该等太久
MAX_SCAN_FILES = 2000
GREP_TIME_BUDGET_SECONDS = 5.0
MAX_SCAN_FILE_BYTES = 2 * 1024 * 1024
MAX_GLOB_MATCHES = 200

# 单行展示上限：压缩的 JSON / 单行日志整行返回毫无意义，
# 而且匹配点落在行尾时简单截断会把它切掉——必须按匹配点居中开窗
GREP_LINE_CHAR_LIMIT = 300

# 用来判断「用户可能想写正则」的元字符。命中时在无结果的情况下给提示，
# 把「静默不命中」变成「可恢复的错误」。
_REGEX_METACHARS = frozenset("^$*+?()[]{}|\\")


class GrepHit(TypedDict):
    path: str
    line: int
    text: str


class GrepOutcome(TypedDict, total=False):
    ok: bool
    code: str
    mode: str
    hits: list[GrepHit]
    # 保序的 (文件, 命中数)，不用 dict 是因为要保持确定性输出
    counts: list[tuple[str, int]]
    files: list[str]
    total_matches: int
    scanned_files: int
    skipped_binary: int
    skipped_large: int
    truncated: bool
    truncated_reason: str
    # 无结果且 pattern 含正则元字符时为 True，提示模型换 regex=True
    regex_hint: bool
    note: str


class GlobOutcome(TypedDict, total=False):
    ok: bool
    code: str
    items: list[dict]
    truncated: bool
    note: str


def _rel_posix(root: Path, target: Path) -> str:
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return target.as_posix()


def _iter_files(base: Path) -> Iterator[Path]:
    """按确定顺序遍历，顺便剪掉 `.git` / `node_modules` 这类目录。

    排序是刻意的：`os.walk` 的顺序依赖文件系统，同一份工作区两次调用可能不同，
    结果不稳定会让「模型上次明明搜到了」变成玄学。
    """
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            yield Path(dirpath) / name


def _match_glob(rel_path: str, pattern: str) -> bool:
    """`glob` 参数的过滤语义。

    含 `/` 的模式按相对路径匹配，否则只比文件名——后者是模型最常用的写法
    （`*.py`），按整路径匹配会让它一个都搜不到，而它并不知道自己错了。
    """
    if "/" in pattern:
        return fnmatch.fnmatch(rel_path, pattern)
    return fnmatch.fnmatch(rel_path.rsplit("/", 1)[-1], pattern)


def _clip_line(line: str, pos: int) -> str:
    """超长行以匹配点为中心开窗。"""
    line = line.rstrip()
    if len(line) <= GREP_LINE_CHAR_LIMIT:
        return line
    half = GREP_LINE_CHAR_LIMIT // 2
    start = max(0, pos - half)
    end = start + GREP_LINE_CHAR_LIMIT
    if end > len(line):
        end = len(line)
        start = max(0, end - GREP_LINE_CHAR_LIMIT)
    piece = line[start:end]
    if start > 0:
        piece = "…" + piece
    if end < len(line):
        piece = piece + "…"
    return piece


def grep_files(
    root: Path,
    pattern: str,
    *,
    path: str = ".",
    glob: str | None = None,
    regex: bool = False,
    mode: str = "content",
    head_limit: int = 50,
) -> GrepOutcome:
    """在工作区里按内容搜索。

    **默认字面匹配而不是正则**：模型写错 `(`、`[`、`$` 是高频事故，而且会
    *静默不命中*——比报错更糟，它会以为文件里没有，然后换别的方法乱试。
    需要正则时显式传 `regex=True`；忘记传但又写了元字符时会得到提示。
    """
    if mode not in ("content", "files_with_matches", "count"):
        return GrepOutcome(ok=False, code="bad_mode")
    if not pattern:
        return GrepOutcome(ok=False, code="empty_pattern")

    try:
        base = safe_resolve(root, path or ".")
    except WorkspaceError as e:
        return GrepOutcome(ok=False, code="bad_path", note=str(e))
    if not base.exists():
        return GrepOutcome(ok=False, code="no_path")

    compiled: re.Pattern[str] | None = None
    if regex:
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return GrepOutcome(ok=False, code="bad_regex", note=str(e))

    limit = max(1, int(head_limit))
    started = time.monotonic()
    hits: list[GrepHit] = []
    counts: list[tuple[str, int]] = []
    matched_files: list[str] = []
    total = 0
    scanned = 0
    skipped_binary = 0
    skipped_large = 0
    truncated = False
    truncated_reason = ""

    candidates: Iterator[Path] = iter([base]) if base.is_file() else _iter_files(base)

    for fp in candidates:
        if scanned >= MAX_SCAN_FILES:
            truncated, truncated_reason = True, f"已达扫描上限 {MAX_SCAN_FILES} 个文件"
            break
        if time.monotonic() - started > GREP_TIME_BUDGET_SECONDS:
            truncated, truncated_reason = True, f"已达时间预算 {GREP_TIME_BUDGET_SECONDS:g}s"
            break

        rel = _rel_posix(root, fp)
        if glob and not _match_glob(rel, glob):
            continue

        try:
            if fp.stat().st_size > MAX_SCAN_FILE_BYTES:
                skipped_large += 1
                continue
            raw = fp.read_bytes()
        except OSError:
            # 权限或竞态：跳过，不能让一个读不了的文件毁掉整次搜索
            continue

        if is_binary(raw):
            skipped_binary += 1
            continue

        scanned += 1
        text = raw.decode("utf-8", errors="replace")
        file_hits = 0

        for line_no, line in enumerate(text.splitlines(), 1):
            if compiled is not None:
                m = compiled.search(line)
                if m is None:
                    continue
                pos = m.start()
            else:
                pos = line.find(pattern)
                if pos < 0:
                    continue

            file_hits += 1
            total += 1

            if mode == "content":
                if len(hits) < limit:
                    hits.append(GrepHit(path=rel, line=line_no, text=_clip_line(line, pos)))
                else:
                    truncated, truncated_reason = True, f"仅显示前 {limit} 条"
                    break
            elif mode == "files_with_matches":
                break

        if file_hits:
            if mode == "files_with_matches":
                matched_files.append(rel)
                if len(matched_files) >= limit:
                    truncated, truncated_reason = True, f"仅显示前 {limit} 个文件"
                    break
            elif mode == "count":
                counts.append((rel, file_hits))

        if mode == "content" and len(hits) >= limit:
            # 命中条目已满，但 total 是「至少这么多」，如实标注 truncated
            truncated, truncated_reason = True, f"仅显示前 {limit} 条"
            break

    has_result = bool(hits or matched_files or counts)
    return GrepOutcome(
        ok=True,
        mode=mode,
        hits=hits,
        counts=counts,
        files=matched_files,
        total_matches=total,
        scanned_files=scanned,
        skipped_binary=skipped_binary,
        skipped_large=skipped_large,
        truncated=truncated,
        truncated_reason=truncated_reason,
        regex_hint=(not has_result and not regex and bool(_REGEX_METACHARS & set(pattern))),
    )


def glob_files(
    root: Path,
    pattern: str,
    *,
    path: str = ".",
    limit: int = MAX_GLOB_MATCHES,
) -> GlobOutcome:
    """按名字模式列出文件（支持 `**`、`*`、`?`、`[abc]`）。"""
    if not pattern:
        return GlobOutcome(ok=False, code="empty_pattern")
    if pattern.startswith(("/", "\\")) or ".." in Path(pattern).parts:
        return GlobOutcome(ok=False, code="bad_pattern", note="模式不能是绝对路径或包含 ..")

    try:
        base = safe_resolve(root, path or ".")
    except WorkspaceError as e:
        return GlobOutcome(ok=False, code="bad_path", note=str(e))
    if not base.exists():
        return GlobOutcome(ok=False, code="no_path")
    if not base.is_dir():
        return GlobOutcome(ok=False, code="not_dir")

    root_resolved = root.resolve()
    items: list[dict] = []
    truncated = False
    # 收集上限：`**` 在超大目录树里能匹配上万条，先多收一点再排序截断，
    # 避免为了「排序正确」把整棵树都拿进内存
    collect_cap = max(limit * 5, 1000)

    try:
        for p in base.glob(pattern):
            if len(items) >= collect_cap:
                truncated = True
                break
            rel = _rel_posix(root, p)
            parts = Path(rel).parts
            # `Path.glob` 会钻进 .git / node_modules，且无法剪枝，只能后过滤
            if any(part in SKIP_DIRS or part.startswith(".") for part in parts[:-1]):
                continue
            try:
                resolved = p.resolve()
            except OSError:
                continue
            if not resolved.is_relative_to(root_resolved):
                # 软链指向工作区外：不列出来，也不要去 stat 它
                continue
            try:
                stat = p.stat()
            except OSError:
                continue
            items.append(
                {
                    "path": rel,
                    "is_dir": p.is_dir(),
                    "size": int(stat.st_size),
                    "mtime": float(stat.st_mtime),
                }
            )
    except (ValueError, OSError) as e:
        return GlobOutcome(ok=False, code="bad_pattern", note=str(e))

    items.sort(key=lambda it: it["path"])
    if len(items) > limit:
        items = items[:limit]
        truncated = True

    return GlobOutcome(ok=True, items=items, truncated=truncated)


__all__ = [
    "MAX_EDIT_BYTES",
    "MAX_GLOB_MATCHES",
    "MAX_READ_BYTES",
    "EditOutcome",
    "GlobOutcome",
    "GrepHit",
    "GrepOutcome",
    "ReadWindow",
    "SKIP_DIRS",
    "detect_newline",
    "edit_text",
    "glob_files",
    "grep_files",
    "is_binary",
    "read_window",
    "strip_line_number_prefix",
]
