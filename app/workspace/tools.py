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
from app.workspace.ops import EditOutcome, edit_text

# 单次列目录返回的条目上限：工作区被写满时，一次 list 也不该把上下文打爆
MAX_LIST_ENTRIES = 200


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"


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
    def workspace_read_file(path: str, max_chars: int = 20000) -> str:
        """读取工作区里的文本文件内容。

        何时使用：需要引用自己之前写入、或用户让你查看的工作区文件时。
        何时不要使用：读会话附件请用 read_session_attachment（附件是只读的，不在工作区里）。
        长文件建议先用 workspace_list_dir 确认大小，再决定是否分段读取。
        path：相对工作区根目录的路径，如 "report.md"。
        max_chars：本次最多返回多少字符，默认 20000，超过会被截断并提示。
        """
        try:
            # schema 已在入口校验类型，这里是二次兜底：直接调 `.func()` 时仍能钳值
            budget = max(1000, min(int(max_chars), 100_000))
        except (TypeError, ValueError):
            budget = 20000
        try:
            text = ws.read_text(path or ".", max_bytes=budget * 3)
        except WorkspaceError as e:
            return f"错误：{e}"
        if len(text) > budget:
            text = text[:budget] + f"\n...[已截断，仅显示前 {budget} 字符，可加大 max_chars 或分段读取]"
        return text or "(文件为空)"

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

    tools: list[Any] = [workspace_list_dir, workspace_read_file, workspace_write_file, workspace_edit_file]

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
                from app.core.turn_runtime import emit_rag_step
                from app.workspace.command_policy import is_safe_readonly_command

                _safe, _why = is_safe_readonly_command(command or "")
                if _safe:
                    emit_rag_step("🔓", "只读命令免确认", f"{(command or '').strip()[:100]}｜{_why}")
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
