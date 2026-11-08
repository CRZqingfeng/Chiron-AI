"""只读命令白名单：判断一条 shell 命令是否可以免人工确认直接执行。

## 为什么需要

`workspace_run_command` 现在是**所有**命令都要用户点一次「允许」，包括 `ls -la`
和 `git diff` 这种纯读操作。日常用起来就是每看一次目录都要点一次——用户会烦到
直接把审批关掉，那反而更危险。

## 设计原则：宁可少放行

判定必须能**证明**这条命令是只读的，证不出来就当危险。四层约束，任一层不过即拒绝：

1. **元字符一律拒绝**：`; & | > < $ ` ( ) { } [ ] # ~ * ? % !` 与换行。
   只要出现，就可能是「一条命令串两条」或「重定向写文件」，不再往下判断。
2. **可执行文件白名单**：只有明确只读的几个二进制（`ls` / `cat` / `git status` …）
   能通过。`python` / `bash` / `curl` / `sed` / `awk` / `find` 一律不放行——
   它们都能一步变出写文件或联网的能力，且难以靠参数限制住。
3. **每个二进制的危险参数拒绝**：例如 `git branch -D`（删分支）、`--output`
   （任意程序写文件）、`wc -f`（读文件）都要挡掉。
4. **路径不许越出工作区**：出现 `..`、绝对路径、盘符、`~` 即拒绝。
   `workspace_read_file` 有 `safe_resolve` 兜底，走 shell 的命令没有，只能靠这里。

## 与审批中间件的分工

本模块只回答「安不安全」，不负责「要不要中断」——那是
`app/chat/approval.SafeCommandAwareHITL` 的事。两处必须用**同一个函数**判定，
否则会出现「工具说免确认、中间件照样拦」或更糟的反向不一致。
"""

from __future__ import annotations

import shlex

# 需要审批的命令工具名（与 settings.CHAT_APPROVAL_TOOLS 的默认值一致）
SAFE_COMMAND_TOOL = "workspace_run_command"

# 只要出现就拒绝的元字符：串命令、重定向、变量展开、通配符、注释
_FORBIDDEN_CHARS = frozenset(";&|><`$(){}[]#~*?%!\n\r\t")

# 只读二进制白名单。故意**不含** python / bash / sh / curl / wget / sed / awk /
# find / xargs / tee / touch / mkdir / rm / mv / cp / dd / chmod / ln / kill。
_SAFE_BINARIES = frozenset(
    {
        "ls",
        "dir",
        "pwd",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "stat",
        "file",
        "du",
        "tree",
        "sort",
        "uniq",
        "cut",
        "diff",
        "git",
    }
)

# git 子命令白名单。刻意排除 branch / remote / config / stash / checkout / reset /
# clean / add / commit / push / pull 等会改状态或改配置的。
_SAFE_GIT_SUBCOMMANDS = frozenset(
    {"status", "diff", "log", "show", "rev-parse", "ls-files", "describe", "blame", "shortlog"}
)

# 任何二进制都拒绝的参数：写文件、执行外部程序、删除
_FORBIDDEN_ARGS = frozenset(
    {"--output", "--exec", "--delete", "-delete", "--in-place", "--remove"}
)

# 每个二进制额外要拒绝的参数。
# 注意不要「一刀切」禁 -o / -i / -f：`grep -i`（忽略大小写）、`grep -o`（只输出匹配）
# 是常用只读参数，误禁的结果是用户体验退化到「什么都得点一次」，那就没必要做这个功能了。
# 只有真正把 -o 当「输出到文件」用的二进制才禁：`sort -o out.txt`、`tree -o out.txt`。
_FORBIDDEN_ARGS_BY_BINARY: dict[str, frozenset[str]] = {
    "sort": frozenset({"-o", "--output"}),
    "tree": frozenset({"-o", "--output"}),
    "git": frozenset({"--output"}),
}

# 路径类参数的拒绝前缀（绝对路径、盘符、家目录）
_ABS_PREFIXES = ("/", "\\", "~")


def _looks_like_escape(token: str) -> bool:
    """参数是否想跳出工作区：`..`、绝对路径、盘符。"""
    if ".." in token:
        return True
    if token.startswith(_ABS_PREFIXES):
        return True
    # Windows 盘符：C:\ 或 C:/
    return len(token) >= 2 and token[1] == ":" and token[0].isalpha()


def is_safe_readonly_command(command: str) -> tuple[bool, str]:
    """判断命令是否**可证明只读**。

    :return: `(True, "命中的二进制名")` 或 `(False, "拒绝原因（给日志/测试看的）")`
    """
    raw = (command or "").strip()
    if not raw:
        return False, "命令为空"
    if len(raw) > 300:
        return False, "命令过长（>300 字符）"

    bad = sorted({c for c in raw if c in _FORBIDDEN_CHARS})
    if bad:
        pretty = " ".join(repr(c) for c in bad)
        return False, f"含元字符 {pretty}（可能是串命令/重定向/变量展开/通配符）"

    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError as e:
        return False, f"命令无法解析：{e}"
    if not tokens:
        return False, "命令为空"

    # 取 basename，兼容 `./ls` 这种写法
    binary = tokens[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if binary not in _SAFE_BINARIES:
        return False, f"`{binary}` 不在只读白名单内"

    args = tokens[1:]

    extra_forbidden = _FORBIDDEN_ARGS_BY_BINARY.get(binary, frozenset())
    for arg in args:
        if _looks_like_escape(arg):
            return False, f"参数 `{arg}` 试图离开工作区"
        low = arg.lower()
        if low in _FORBIDDEN_ARGS or low in extra_forbidden:
            return False, f"参数 `{arg}` 会写文件或执行外部程序"
        # `--output=/tmp/x` 这种挂等号的写法也要挡
        if any(low.startswith(x + "=") for x in ("--output", "--exec", "--in-place")):
            return False, f"参数 `{arg}` 会写文件或执行外部程序"

    if binary == "git":
        if not args:
            return False, "git 未给子命令"
        sub = args[0].lower()
        if sub.startswith("-"):
            # `git --version` 之类：只允许显式版本查询
            if sub in ("--version", "-v"):
                return True, "git --version"
            return False, f"git 全局参数 `{args[0]}` 未放行"
        if sub not in _SAFE_GIT_SUBCOMMANDS:
            return False, f"git 子命令 `{sub}` 不在只读白名单内（可能改状态）"

    return True, binary


def describe_decision(command: str) -> str:
    """给日志/思考区用的一句话结论。"""
    ok, reason = is_safe_readonly_command(command)
    return f"免确认（只读：{reason}）" if ok else f"需确认（{reason}）"


# --- 测试类命令（准只读）--------------------------------------------------

# 测试 / 静态校验的调用前缀。
#
# **为什么不并进只读白名单**：pytest 会加载 `conftest.py`，而那是模型自己可写的文件——
# 一旦把 pytest 当成只读免审批，`写 conftest.py` + `pytest` 就成了绕过
# `python script.py` 审批的通道。所以它必须挂在「本会话已授权」这个前置条件下，
# 由 `app.chat.approval` 统一判定。
_TEST_INVOCATIONS: tuple[tuple[str, ...], ...] = (
    ("pytest",),
    ("py.test",),
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
    ("python", "-m", "py_compile"),
    ("python3", "-m", "py_compile"),
    ("python", "-m", "unittest"),
    ("python3", "-m", "unittest"),
    ("python", "-m", "json.tool"),
    ("python3", "-m", "json.tool"),
)


def is_test_command(command: str) -> tuple[bool, str]:
    """判断是否「测试 / 静态校验」类命令。

    这类命令**不是**可证明只读：pytest 会写 `.pytest_cache`、`__pycache__`。
    但它的风险比任意脚本低得多，值得在用户批准一次之后复用。

    :return: `(True, "命中的调用形式")` 或 `(False, "原因")`
    """
    raw = (command or "").strip()
    if not raw or len(raw) > 300:
        return False, "命令为空或过长"
    if any(c in _FORBIDDEN_CHARS for c in raw):
        return False, "含元字符"
    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError as e:
        return False, f"命令无法解析：{e}"
    if not tokens:
        return False, "命令为空"

    for prefix in _TEST_INVOCATIONS:
        if tuple(tokens[: len(prefix)]) != prefix:
            continue
        # 路径参数不能跳出工作区；`-` 开头的是选项，跳过
        for tok in tokens[len(prefix) :]:
            if not tok.startswith("-") and _looks_like_escape(tok):
                return False, f"参数疑似越界：{tok}"
        return True, " ".join(prefix)
    return False, "不是测试类命令"
