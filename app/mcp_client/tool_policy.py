"""高危 MCP 工具的**判定**（不负责执行）。

## 这个模块只管一件事：这个工具要不要人工确认

执行方式由 `app/chat/approval.SafeCommandAwareHITL` 统一处理（U5c）：
判定命中的工具名会进 `interrupt_on`，由 LangGraph 的断点中断拦住，
用户点「允许」后图**从中断点继续**，已执行过的步骤不重跑。

## 历史：这里曾经有一套「假审批」

U5c 之前，本模块用 `wrap_mcp_tool_with_confirmation` 把每个高危工具包一层：
调用时先查 Redis 票据，没有就返回一句 `MCP_CONFIRMATION_REQUIRED pending_id=...`
字符串，靠模型读到后停下来问用户；用户确认后前端**新开一个 job 整轮重跑**，
再靠一段注入的提示文本诱导模型重新调用一次。

那套做法的代价（实测）：
- 中断前已执行的读类工具会在重跑里**再执行一遍**（token + 外部 API 配额双浪费）
- 票据 TTL 只有 300s，用户离开十分钟回来点确认必然失效
- 两套卡片两套状态，前端 `ChatMessageItem.vue` 要维护两份模板

## 判定顺序（改这里的人请注意）

1. `MCP_CONFIRMATION_REQUIRED=false` → 一律不确认（全局总开关）
2. 服务级 `confirm_policy`：`never` 不确认 / `always` 全确认 / `auto` 走向下判定
3. `auto` 时看 MCP 的 annotations：`readOnlyHint=true` → 放行；`destructiveHint=true` → 拦截
4. 都没有标注时，按「服务名 + 工具名 + 描述」里的动词猜（`create` / `delete` …）

第 4 步是启发式，一定会有误判。它的存在意义是**默认保守**：
拿不准的时候问一句，代价远小于误执行一次写操作。
"""

from __future__ import annotations

import re

from app.settings import settings

_WRITE_ACTIONS = {
    "create",
    "update",
    "delete",
    "remove",
    "merge",
    "push",
    "invite",
    "publish",
    "write",
    "secret",
    "deploy",
    "execute",
    "add",
    "assign",
    "close",
    "reopen",
    "lock",
    "unlock",
    "transfer",
    "fork",
}


def _tokenize(text: str) -> set[str]:
    """拆成小写词集合，驼峰先按大小写边界切开（`createIssue` → `create issue`）。"""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _write_actions() -> set[str]:
    return set(_WRITE_ACTIONS)


def confirmation_required() -> bool:
    return bool(getattr(settings, "MCP_CONFIRMATION_REQUIRED", True))


def _looks_like_annotations(value: dict) -> bool:
    """这个 dict 本身像不像 MCP annotations（而不是装着 annotations 的容器）。"""
    return any(
        k in value
        for k in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")
    )


def _tool_annotations(tool: object) -> dict:
    """从工具对象上挖 MCP annotations。

    **实测的四种形态都要认**（`langchain-mcp-adapters` 各版本不同）：

    1. `tool.annotations` 本身就是 `{"readOnlyHint": True}`
    2. `tool.metadata` **平铺**——adapters 0.3.x 的做法：
       `metadata = {**tool.annotations.model_dump(), "_meta": ...}`（见 `tools.py:510-513`）
    3. `tool.metadata["annotations"]` 嵌套
    4. `tool.metadata["mcp"]["annotations"]` 嵌套更深

    第 2 种是 U5c 补的。之前只认 3/4，于是**所有 `readOnlyHint` 声明都被丢掉**，
    只读工具只能靠名字里的动词猜——表现为「偶尔多弹一次确认」，很难被注意到。

    挖不到就返回空 dict，退化成按关键词猜（保守方向）。
    """
    candidates = (
        getattr(tool, "annotations", None),
        getattr(tool, "metadata", None),
        getattr(tool, "extra", None),
    )
    for value in candidates:
        if not isinstance(value, dict):
            continue
        if isinstance(value.get("annotations"), dict):
            return value["annotations"]
        if isinstance(value.get("mcp"), dict) and isinstance(value["mcp"].get("annotations"), dict):
            return value["mcp"]["annotations"]
        if _looks_like_annotations(value):
            return value
    return {}


def _ann_bool(annotations: dict, key: str) -> bool | None:
    value = annotations.get(key)
    return value if isinstance(value, bool) else None


def is_high_risk_mcp_tool(server_name: str, tool_name: str, description: str, annotations: dict | None = None) -> bool:
    """按 annotations 优先、关键词兜底的顺序判定是否高危。"""
    annotations = annotations or {}
    read_only = _ann_bool(annotations, "readOnlyHint")
    destructive = _ann_bool(annotations, "destructiveHint")
    if read_only is True:
        return False
    if destructive is True:
        return True
    tokens = _tokenize(f"{server_name} {tool_name} {description}")
    return bool(tokens & _write_actions())


def should_confirm_mcp_tool(server_name: str, tool: object, confirm_policy: str = "auto") -> bool:
    """这个工具要不要人工确认。

    :param tool: **原始的** MCP 工具对象。不要在 `_guard_mcp_tool_output` 之后再传——
        那一步用 `StructuredTool.from_function` 重建了工具，annotations 会丢，
        只读工具会被误判成高危。
    """
    if not confirmation_required():
        return False
    policy = (confirm_policy or "auto").strip().lower()
    if policy == "never":
        return False
    if policy == "always":
        return True
    tool_name = getattr(tool, "name", None) or "mcp_tool"
    description = getattr(tool, "description", "") or ""
    return is_high_risk_mcp_tool(server_name, tool_name, description, _tool_annotations(tool))
