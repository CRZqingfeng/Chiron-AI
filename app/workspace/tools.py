"""工作区工具：给模型用的三个文件操作入口。

`execute` 刻意**不在这里注册**——见 `docs/dev/U4-思路.md` 发现二：
在没有审批中断（U5b）之前，模型不应具备执行任意 shell 的能力。
接口已经实现且有测试，U5b 只需在此加一个 `@tool` 包装 + 接 interrupt。

风格跟随现有工具（`attachment_tools.py`）：描述里必须写清
「何时使用 / 何时不要使用 / 参数从哪来」，否则模型不会正确选择。
出错时返回中文错误字符串而不是抛异常——工具抛异常会让整轮生成直接失败。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.workspace.base import BaseWorkspace, WorkspaceError
from app.workspace.ops import (
    EditOutcome,
    GlobOutcome,
    GrepOutcome,
    ReadWindow,
    edit_text,
    glob_files,
    grep_files,
    read_window,
)

# 单次列目录返回的条目上限：工作区被写满时，一次 list 也不该把上下文打爆
MAX_LIST_ENTRIES = 200


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"


def _format_read_result(path: str, out: ReadWindow) -> str:
    """按「行号 | 内容」渲染，并在末尾给出**可执行的续读指令**。

    旧版的截断提示写着「可加大 max_chars 或分段读取」，但当时没有 offset，
    分段读取根本做不到——模型会照着提示重试，然后撞同一个上限。
    现在提示里直接给 `offset=N`，它抄过去就能用。
    """
    code = out.get("code", "")
    if code == "bad_path":
        return f"错误：{out.get('note') or '路径不合法'}。"
    if code == "no_file":
        return f"错误：文件不存在：{path}。先用 workspace_list_dir 确认路径。"
    if code == "is_dir":
        return f"错误：{path} 是目录，请用 workspace_list_dir。"
    if code == "too_large":
        size_mb = int(out.get("note") or 0) / 1024 / 1024
        return f"错误：文件过大（{size_mb:.1f}MB），单次最多读 5MB。"
    if code == "not_text":
        return f"错误：{path} 不是 UTF-8 文本文件，读不了。"
    if code == "offset_out_of_range":
        total = out.get("total_lines", 0)
        return f"错误：offset 超出范围（文件共 {total} 行）。用 offset=1 从头读。"

    rows = out.get("lines") or []
    if not rows:
        return "(文件为空)"

    total = out.get("total_lines", 0)
    width = max(3, len(str(total)))
    body = "\n".join(f"{n:>{width}} | {text}" for n, text in rows)

    end = out.get("end_line", 0)
    start = out.get("start_line", 1)

    # 单行超长必须单独提示：即使没有「更多行」，模型也得知道自己拿到的是残行，
    # 否则它会拿半截内容去做 edit，匹配永远失败。
    if out.get("stop_reason") == "long_line":
        return (
            f"{body}\n...（第 {end} 行本身超过单次长度上限，已截断显示；"
            f"这一行建议用 workspace_grep 定位关键片段，或用 run_command 处理）"
        )

    if not out.get("has_more"):
        return body

    if out.get("stop_reason") == "chars":
        tail = f"...（单次长度已达上限，停在第 {end} 行；共 {total} 行，用 offset={end + 1} 继续读）"
    else:
        tail = f"...（已显示第 {start}-{end} 行，共 {total} 行；用 offset={end + 1} 继续读）"
    return f"{body}\n{tail}"


def _format_edit_error(path: str, out: EditOutcome) -> str:
    """把 `edit_text` 的失败码翻译成**可执行的中文建议**。

    为什么每条都带下一步动作：只说「未找到」模型会原地重试同一个参数，
    说「先用 read_file 确认原文」它才会换策略。错误文案是 ACI 的一部分。
    """
    code = out.get("code", "")
    if code == "bad_path":
        return f"错误：{out.get('note') or '路径不合法'}。"
    if code == "no_file":
        return f"错误：文件不存在：{path}。先用 workspace_list_dir 确认路径。"
    if code == "is_dir":
        return f"错误：{path} 是目录，请用 workspace_list_dir。"
    if code == "too_large":
        size_mb = int(out.get("note") or 0) / 1024 / 1024
        return f"错误：文件过大（{size_mb:.1f}MB），edit_file 仅支持 2MB 以内的文本文件。"
    if code == "not_text":
        return f"错误：{path} 不是 UTF-8 文本文件，无法编辑。"
    if code == "empty_old":
        return "错误：old_string 不能为空。"
    if code == "same":
        return "错误：old_string 与 new_string 相同，无需修改。"
    if code == "unsafe_replace_all":
        return "错误：replace_all=True 时 old_string 不能是纯空白。"
    if code == "write_failed":
        return f"错误：写入失败（{out.get('note') or '未知原因'}），工作区可能只读或磁盘已满。"
    if code == "not_found":
        return (
            f"错误：未找到匹配内容（文件共 {out.get('total_lines', 0)} 行）。"
            "请先用 workspace_read_file 读取原文，逐字复制要替换的片段（含缩进与空行）。"
        )
    if code == "ambiguous":
        lines = out.get("match_lines") or []
        shown = "、".join(str(n) for n in lines[:5])
        more = " 等" if len(lines) > 5 else ""
        return (
            f"错误：找到 {out.get('occurrences', 0)} 处匹配（第 {shown}{more} 行）。"
            "请提供更长的上下文使其唯一，或传 replace_all=True 全部替换。"
        )
    return f"错误：编辑失败（{code}）。"


def _format_edit_ok(path: str, out: EditOutcome) -> str:
    occurrences = out.get("occurrences", 1)
    first_line = out.get("first_line", 1)
    total_lines = out.get("total_lines", 0)
    newline = out.get("newline", "LF")
    before = out.get("before") or []
    after = out.get("after") or []

    head = f"已修改 {path}：第 {first_line} 行（{occurrences} 处，文件共 {total_lines} 行，行尾 {newline}）"
    note = out.get("note") or ""
    if note:
        head += f"\n（{note}）"

    width = max([len(str(n)) for n, _ in before + after] or [1])
    parts = [head, "--- 改动前 ---"]
    parts.extend(f"{n:>{width}} | {text}" for n, text in before)
    parts.append("--- 改动后 ---")
    parts.extend(f"{n:>{width}} | {text}" for n, text in after)
    return "\n".join(parts)


def _format_grep_result(pattern: str, out: GrepOutcome) -> str:
    """把检索结果压成模型能一眼看懂的文本。"""
    code = out.get("code", "")
    if code == "bad_path":
        return f"错误：{out.get('note') or '路径不合法'}。"
    if code == "no_path":
        return "错误：目录不存在，先用 workspace_list_dir 确认路径。"
    if code == "bad_mode":
        return "错误：mode 只能是 content / files_with_matches / count。"
    if code == "empty_pattern":
        return "错误：pattern 不能为空。"
    if code == "bad_regex":
        return f"错误：正则表达式无效（{out.get('note')}）。请修正或改用字面匹配。"

    mode = out.get("mode", "content")
    scanned = out.get("scanned_files", 0)
    skipped_binary = out.get("skipped_binary", 0)
    skipped_large = out.get("skipped_large", 0)
    truncated = out.get("truncated", False)
    reason = out.get("truncated_reason", "")

    lines: list[str] = []

    if mode == "content":
        hits = out.get("hits") or []
        if not hits:
            lines.append(f'在工作区里没有找到 "{pattern}"（扫描了 {scanned} 个文件）。')
        else:
            files = len({h["path"] for h in hits})
            lines.append(f'在 {files} 个文件里找到 {len(hits)} 处匹配 "{pattern}"：')
            lines.extend(f'{h["path"]}:{h["line"]}: {h["text"]}' for h in hits)
    elif mode == "files_with_matches":
        files = out.get("files") or []
        if not files:
            lines.append(f'没有文件包含 "{pattern}"（扫描了 {scanned} 个文件）。')
        else:
            lines.append(f'{len(files)} 个文件包含 "{pattern}"：')
            lines.extend(files)
    else:  # count
        counts = out.get("counts") or []
        if not counts:
            lines.append(f'没有文件包含 "{pattern}"（扫描了 {scanned} 个文件）。')
        else:
            total = sum(c for _, c in counts)
            lines.append(f'{len(counts)} 个文件共命中 {total} 处：')
            lines.extend(f"{path}: {c}" for path, c in counts)

    if truncated and reason:
        lines.append(f"（{reason}，结果不完整；可缩小 path 或加 glob 过滤）")

    skipped: list[str] = []
    if skipped_binary:
        skipped.append(f"{skipped_binary} 个二进制文件")
    if skipped_large:
        skipped.append(f"{skipped_large} 个超过 2MB 的文件")
    if skipped:
        lines.append("（已跳过：" + "、".join(skipped) + "）")

    if out.get("regex_hint"):
        lines.append("提示：pattern 里含正则元字符，但当前是字面匹配。需要正则请传 regex=True。")

    return "\n".join(lines)


def _format_glob_result(pattern: str, out: GlobOutcome) -> str:
    code = out.get("code", "")
    if code == "bad_path":
        return f"错误：{out.get('note') or '路径不合法'}。"
    if code == "no_path":
        return "错误：目录不存在，先用 workspace_list_dir 确认路径。"
    if code == "not_dir":
        return "错误：path 需要是目录。"
    if code == "empty_pattern":
        return "错误：pattern 不能为空。"
    if code == "bad_pattern":
        return f"错误：模式无效（{out.get('note') or '不支持绝对路径或 ..'}）。"

    items = out.get("items") or []
    if not items:
        return f"没有匹配 {pattern} 的文件。"

    lines = [f"匹配 {pattern} 共 {len(items)} 项："]
    for it in items:
        kind = "[目录]" if it["is_dir"] else f"({_fmt_size(it['size'])})"
        lines.append(f"{it['path']}  {kind}")
    if out.get("truncated"):
        lines.append("（结果已截断，请缩小 pattern 或指定 path）")
    return "\n".join(lines)


def make_workspace_tools(ws: BaseWorkspace, *, include_exec: bool = False) -> list[Any]:
    """基于给定工作区实例创建工具闭包。

    工作区随 job 创建、随 job 销毁，所以工具也是每轮新建——
    与 `build_model_and_agent` 每轮重建的节奏一致。

    :param include_exec: 是否包含 `workspace_run_command`。**默认 False**，
        只在调用方确认「审批中断已生效」时才传 True——命令执行能力必须与
        人工确认同时到位，否则等于把 `rm -rf` 的按钮直接交给模型。
    """

    @tool
    def workspace_list_dir(path: str = ".") -> str:
        """列出工作区某个目录下的文件与子目录（只列一层，不递归）。

        何时使用：不確定工作区里已有哪些文件时先列一次；或用户问「工作区里有什么」。
        何时不要使用：已知具体文件路径时直接 read_file，不要为了确认存在而多调一次。
        path：相对工作区根目录的路径，默认 "." 表示根目录；不要传绝对路径。
        """
        try:
            items = ws.list_dir(path or ".")
        except WorkspaceError as e:
            return f"错误：{e}"
        if not items:
            return f"目录 {path or '.'} 为空。"
        lines = [f"目录 {path or '.'} 下共 {len(items)} 项（目录在前）："]
        for it in items[:MAX_LIST_ENTRIES]:
            kind = "目录" if it["is_dir"] else _fmt_size(int(it["size"]))
            lines.append(f"- {it['path']}  [{kind}]")
        if len(items) > MAX_LIST_ENTRIES:
            lines.append(f"...其余 {len(items) - MAX_LIST_ENTRIES} 项已省略")
        return "\n".join(lines)

    @tool
    def workspace_read_file(
        path: str,
        offset: int = 1,
        limit: int = 2000,
        max_chars: int = 20000,
    ) -> str:
        """读取工作区里的文本文件（输出带行号，可分页续读）。

        何时使用：需要引用自己之前写入、或用户让你查看的工作区文件时；
                    改文件之前也先读一遍，确认原文到底长什么样。
        何时不要使用：读会话附件请用 read_session_attachment（附件不在工作区里）；
                       想找「哪个文件里有某段内容」用 workspace_grep，不要逐个读文件。
        path：相对工作区根目录的路径，如 "report.md"。
        offset：从第几行开始读，**行号从 1 开始**（不是 0），默认 1。
        limit：本次最多读多少行，默认 2000。
        max_chars：单次返回的字符上限，默认 20000；超出会提前停并在末尾告诉你从哪一行继续。

        注意：每行前面的「行号 |」是给你定位用的，**不属于文件内容**——
        用 workspace_edit_file 时 old_string 不要带这个前缀。
        """
        out = read_window(
            ws.root_path,
            path or "",
            offset=offset,
            limit=limit,
            max_chars=max_chars,
        )
        return _format_read_result(path or "", out)

    @tool
    def workspace_write_file(path: str, content: str) -> str:
        """把文本写入工作区文件（覆盖同名文件，父目录自动创建）。

        何时使用：需要生成报告、整理资料、保存中间结果供后续轮次使用时。
        何时不要使用：不要用它修改会话附件（附件只读）；单文件建议不超过 200KB。
        path：相对工作区根目录的路径，如 "report.md" 或 "notes/a.md"；不要传绝对路径。
        content：要写入的完整文本（是覆盖而不是追加）。
        """
        try:
            n = ws.write_text(path or "", content or "")
        except WorkspaceError as e:
            return f"错误：{e}"
        return f"已写入 {path}（{_fmt_size(n)}）。"

    @tool
    def workspace_edit_file(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """精确修改工作区文件里的一小段内容（只替换 old_string，不重写全文）。

        何时使用：文件已存在、只想改动其中几行时。**改现有文件优先用它**，不要用
        workspace_write_file 重写全文——重写长文件容易漏内容，漏掉的部分还不会报错。
        何时不要使用：新建文件用 workspace_write_file；不知道原文长什么样时，
        先用 workspace_read_file 读一遍，再从中复制片段。
        path：相对工作区根目录的路径，如 "notes/a.md"；不要传绝对路径。
        old_string：要替换的原文片段，必须与文件中逐字一致（含缩进与空行），
                    默认必须在文件里唯一出现；不唯一时请带上更多上下文使其唯一。
        new_string：替换后的内容，不能与 old_string 相同。
        replace_all：为 True 时替换全部匹配（默认 False，要求唯一匹配）。
        """
        out = edit_text(
            ws.root_path,
            path or "",
            old_string or "",
            new_string or "",
            replace_all=bool(replace_all),
        )
        if out.get("ok"):
            return _format_edit_ok(path or "", out)
        return _format_edit_error(path or "", out)

    @tool
    def workspace_grep(
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        mode: str = "content",
        regex: bool = False,
        head_limit: int = 50,
    ) -> str:
        """在工作区文件内容里搜索，找出"哪个文件里有这段内容"。

        何时使用：不确定某个字符串/函数名/配置项出现在哪个文件时。**不要**为了找它
        逐个 workspace_read_file——那会烧掉大量上下文，还容易漏。
        何时不要使用：找"文件名"匹配用 workspace_glob；已知路径时直接 read_file。
        pattern：要搜索的文本。**默认按字面匹配，不是正则**——需要正则请传 regex=True。
                 常用写法：函数名 "compute_total"、配置项 "TIMEOUT"、注释 "TODO"。
        path：从哪个目录开始搜，默认工作区根目录。
        glob：只搜文件名匹配该模式的文件，如 "*.py"、"src/**/*.ts"。
        mode：content（带行号与内容）/ files_with_matches（只列文件）/ count（每个文件命中数）。
        regex：为 True 时把 pattern 当正则表达式。
        head_limit：最多返回多少条（默认 50）。
        """
        try:
            limit = max(1, min(int(head_limit), 200))
        except (TypeError, ValueError):
            limit = 50
        out = grep_files(
            ws.root_path,
            pattern or "",
            path=path or ".",
            glob=glob or None,
            regex=bool(regex),
            mode=(mode or "content"),
            head_limit=limit,
        )
        return _format_grep_result(pattern or "", out)

    @tool
    def workspace_glob(pattern: str, path: str = ".") -> str:
        """按文件名模式列出工作区里的文件（支持 `**` 递归）。

        何时使用：不确定有哪些文件、或需要"所有 .py / 所有 test_*.md"这类清单时。
        何时不要使用：找文件**内容**用 workspace_grep；只需知道某个目录里有什么用
        workspace_list_dir（更省）。
        pattern：glob 模式，如 "*.md"、"**/*.py"、"src/**/*.ts"。
                 不能是绝对路径，也不能包含 ".."。
        path：从哪个目录开始找，默认工作区根目录。
        """
        out = glob_files(ws.root_path, pattern or "", path=path or ".")
        return _format_glob_result(pattern or "", out)

    tools: list[Any] = [
        workspace_list_dir,
        workspace_read_file,
        workspace_write_file,
        workspace_edit_file,
        workspace_grep,
        workspace_glob,
    ]

    if include_exec:

        @tool
        def workspace_run_command(command: str, cwd: str | None = None, timeout: int = 30) -> str:
            """在工作区（隔离沙箱）里执行一条 shell 命令，返回输出。

            何时使用：需要运行脚本、批量处理文件、转换格式、或验证代码能否跑通时。
            何时不要使用：只是读写单个文件请用 workspace_read_file / workspace_write_file；
            需要访问外网请用 fetch_url（沙箱默认无网络）。
            **会请用户确认**：涉及写文件、删改、执行代码的命令。可证明只读的命令
            （ls / cat / head / tail / wc / grep / git status|diff|log 等）免确认直接执行，
            因此不需要为了「看一眼目录」绕路。
            **跑测试**（pytest / python -m py_compile）：本会话批准过一次后免确认。
            ⚠️ 但**不要加管道或重定向**（`| head`、`2>&1`、`>` 都不行）——那样会被当成
            任意命令重新要求确认。需要限制输出就用 `-q`、`--tb=short` 这类参数，
            例如 `python -m pytest -q --tb=short`。
            command：完整的 shell 命令，如 "python script.py" 或 "ls -la"。
            cwd：相对工作区根目录的工作目录，默认根目录。
            timeout：超时秒数，默认 30，最大 120。
            """
            try:
                budget = max(1, min(int(timeout), 120))
            except (TypeError, ValueError):
                budget = 30
            # 免确认时告知用户「为什么没弹卡片」：否则他会以为审批坏了。
            # 判定与中间件共用同一个函数，不会出现「工具说免确认、界面还在等确认」。
            try:
                from app.core.turn_runtime import emit_rag_step, get_session_scope
                from app.workspace.command_grants import grant_test_commands, test_commands_granted
                from app.workspace.command_policy import is_safe_readonly_command, is_test_command

                _cmd = (command or "").strip()
                _safe, _why = is_safe_readonly_command(_cmd)
                if _safe:
                    emit_rag_step("🔓", "只读命令免确认", f"{_cmd[:100]}｜{_why}")
                else:
                    # 能走到这里说明审批已通过（或本来就不需要审批）。
                    # 把「本会话允许跑测试」记下来，下次同类命令不再打断用户。
                    _is_test, _kind = is_test_command(_cmd)
                    if _is_test:
                        _scope = get_session_scope()
                        if not test_commands_granted(_scope) and grant_test_commands(_scope):
                            emit_rag_step(
                                "🔓",
                                "测试命令本会话免确认",
                                f"{_cmd[:100]}｜已批准一次（{_kind}），后续同类命令不再询问",
                            )
            except Exception:  # noqa: BLE001 - 提示失败不影响执行
                pass

            try:
                r = ws.execute(command or "", cwd=cwd or None, timeout=budget)
            except WorkspaceError as e:
                return f"错误：{e}"
            parts = [f"退出码：{r['exit_code']}（耗时 {r['duration_ms']}ms）"]
            if r["stdout"]:
                parts.append(f"--- stdout ---\n{r['stdout']}")
            if r["stderr"]:
                parts.append(f"--- stderr ---\n{r['stderr']}")
            if r["truncated"]:
                parts.append("（输出已截断）")
            if not r["stdout"] and not r["stderr"]:
                parts.append("（无输出）")
            return "\n".join(parts)

        tools.append(workspace_run_command)

    return tools


__all__ = ["make_workspace_tools", "MAX_LIST_ENTRIES"]
