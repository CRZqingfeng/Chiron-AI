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

# 单次列目录返回的条目上限：工作区被写满时，一次 list 也不该把上下文打爆
MAX_LIST_ENTRIES = 200


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"


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

    tools: list[Any] = [workspace_list_dir, workspace_read_file, workspace_write_file]

    if include_exec:

        @tool
        def workspace_run_command(command: str, cwd: str | None = None, timeout: int = 30) -> str:
            """在工作区（隔离沙箱）里执行一条 shell 命令，返回输出。

            何时使用：需要运行脚本、批量处理文件、转换格式、或验证代码能否跑通时。
            何时不要使用：只是读写单个文件请用 workspace_read_file / workspace_write_file；
            需要访问外网请用 fetch_url（沙箱默认无网络）。
            该工具属于高危操作，执行前会请用户确认一次。
            command：完整的 shell 命令，如 "python script.py" 或 "ls -la"。
            cwd：相对工作区根目录的工作目录，默认根目录。
            timeout：超时秒数，默认 30，最大 120。
            """
            try:
                budget = max(1, min(int(timeout), 120))
            except (TypeError, ValueError):
                budget = 30
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
