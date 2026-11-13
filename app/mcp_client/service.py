"""MCP 服务连接与工具加载（超时容错 + 工具名命名空间）。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re

from app.chat.tool_registry import BUILTIN_TOOL_NAMES as _BUILTIN_TOOL_NAMES
from app.chat.tool_registry import RESERVED_TOOL_NAMES as _RESERVED_TOOL_NAMES
from app.mcp_client.tool_policy import should_confirm_mcp_tool
from typing import Any

MCP_CONNECT_TIMEOUT_SECONDS = 12
MCP_TOOL_SCHEMA_TTL_SECONDS = 120

# 模型侧函数名的硬上限（OpenAI 与 Anthropic 都是 64）。超过就截断 + 哈希。
MAX_EXPOSED_NAME_LEN = 64

# 命名空间分隔符：双下划线。
# 不用 `.` 是因为部分 provider 拒绝函数名里出现点号；不用 `:` 同理。
# 反解用 `rsplit("__", 1)`——server 名里即使有下划线也能正确切出工具名。
NAMESPACE_SEPARATOR = "__"

logger = logging.getLogger(__name__)

# 关于 `_BUILTIN_TOOL_NAMES`：内置工具名集合，MCP 工具与之撞名时跳过（保留内置实现）。
#
# 这里**不再自己维护清单**——原先硬编码的那份漏了 4 个 workspace_* 工具，
# 导致 MCP 可以顶替内置实现且不报错（见 app/chat/tool_registry.py 的说明）。
# 现由注册表派生，新增内置工具只需在 TOOL_SPECS 加一行。


def _sanitize_name_part(value: str) -> str:
    """只保留模型函数名允许的字符，其余压成下划线。"""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "").strip()).strip("_")


def mcp_exposed_name(server_name: str, tool_name: str, *, server_id: int | None = None) -> str:
    """MCP 工具**对模型暴露的名字**：`{server}__{tool}`。

    ## 为什么必须加前缀

    不同 MCP server 提供同名工具是常态：Linear 与 Jira 都有 `create_issue`，
    GitHub 与 GitLab 都有 `create_merge_request`。不加前缀时第二个只能丢弃
    （`seen_names` 冲突就 `continue`），用户会**永久失去**其中一半能力，
    而界面上只留一行「工具名冲突，已跳过」。连的 server 越多，丢得越多。

    这与本项目自己的原则也是矛盾的：`tool_registry.assert_no_duplicates` 明确写着
    「重名必须抛错，不能静默保留其一」。内置工具重名抛错，MCP 重名默默丢弃，
    同一件事两套标准。

    ## 名字怎么拼

    - 非 `[A-Za-z0-9_-]` 字符一律压成 `_`（MCP 工具名本身通常是英文，
      但 server 名是用户起的，可能是中文或带空格）
    - server 部分净化后为空（例如纯中文名）→ 退化用 `server{id}`
    - 总长超过 `MAX_EXPOSED_NAME_LEN` 时截断并追加 8 位内容哈希。
      **必须带哈希**：单纯截断会让两个长名工具撞到同一个名字，
      把「丢弃一个」变成「两个都指向错的那个」，比原问题更隐蔽
    """
    server_part = _sanitize_name_part(server_name)
    if not server_part:
        server_part = f"server{server_id}" if server_id is not None else "mcp"
    tool_part = _sanitize_name_part(tool_name) or "tool"

    combined = f"{server_part}{NAMESPACE_SEPARATOR}{tool_part}"
    if len(combined) <= MAX_EXPOSED_NAME_LEN:
        return combined

    digest = hashlib.sha1(combined.encode("utf-8")).hexdigest()[:8]
    keep = MAX_EXPOSED_NAME_LEN - len(digest) - 1  # 留出 `_` 与哈希
    return f"{combined[:keep]}_{digest}"


def _dedupe_exposed_name(name: str, taken: set[str]) -> str:
    """净化或截断后仍撞名时，加内容哈希后缀。

    同一个 server 内部重名属于 server 自身的 bug，但**不能让用户为此少一个工具**，
    所以这里做的是"共存"而不是"丢弃"。
    """
    if name not in taken:
        return name
    for salt in range(1, 100):
        digest = hashlib.sha1(f"{name}:{salt}".encode("utf-8")).hexdigest()[:6]
        keep = MAX_EXPOSED_NAME_LEN - len(digest) - 1
        candidate = f"{name[:keep]}_{digest}"
        if candidate not in taken:
            return candidate
    # 100 次都撞说明 `taken` 被污染了，不是正常路径。用 sha1 而不是内置 hash：
    # 后者对 str 有 PYTHONHASHSEED 随机化，同一份数据每次进程启动结果都不同。
    tail = hashlib.sha1(name.encode("utf-8")).hexdigest()[:7]
    return f"{name[:56]}_{tail}"


def decrypt_headers(headers_ciphertext: str | None) -> dict[str, str]:
    """解密请求头 JSON；异常或为空时返回空 dict。"""
    from app.utils.api_key_crypto import decrypt_api_key_safe

    plain = decrypt_api_key_safe(headers_ciphertext)
    if not plain:
        return {}
    try:
        data = json.loads(plain)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def encrypt_headers(headers: dict[str, str] | None) -> str | None:
    """加密请求头 dict 为密文；空 dict 返回 None。"""
    from app.utils.api_key_crypto import encrypt_api_key

    if not headers:
        return None
    cleaned = {str(k).strip(): str(v) for k, v in headers.items() if str(k).strip()}
    if not cleaned:
        return None
    return encrypt_api_key(json.dumps(cleaned, ensure_ascii=False))


def mcp_tool_schema_cache_key(agent_id: int, server_id: int) -> str:
    return f"mcp_tool_schema:{int(agent_id)}:{int(server_id)}"


def invalidate_mcp_tool_schema_cache(agent_id: int, server_id: int) -> None:
    from app.core.cache import cache

    cache.delete(mcp_tool_schema_cache_key(agent_id, server_id))


def _tool_args_json_schema(tool: Any) -> dict[str, Any]:
    schema_cls = getattr(tool, "args_schema", None)
    if schema_cls is not None:
        try:
            if hasattr(schema_cls, "model_json_schema"):
                out = schema_cls.model_json_schema()
                if isinstance(out, dict):
                    return out
            if hasattr(schema_cls, "schema"):
                out = schema_cls.schema()
                if isinstance(out, dict):
                    return out
        except Exception:
            pass
    args = getattr(tool, "args", None)
    if isinstance(args, dict):
        return {"type": "object", "properties": args}
    return {"type": "object", "properties": {}}


def _snapshot_mcp_tools(tools: list) -> list[dict[str, Any]]:
    snaps: list[dict[str, Any]] = []
    for t in tools:
        snaps.append(
            {
                "name": getattr(t, "name", "") or "",
                "description": getattr(t, "description", "") or "",
                "args_schema": _tool_args_json_schema(t),
            }
        )
    return snaps


def _frozen_args_schema(json_schema: dict | None, *, tool_name: str):
    from pydantic import BaseModel, ConfigDict

    schema = dict(json_schema) if isinstance(json_schema, dict) else {"type": "object", "properties": {}}

    class Frozen(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return schema

    safe = "".join(c if c.isalnum() else "_" for c in (tool_name or "mcp")) or "mcp"
    Frozen.__name__ = f"{safe}StubArgs"
    Frozen.__qualname__ = Frozen.__name__
    return Frozen


def _stub_mcp_tools_from_snapshot(snaps: list, server_name: str, error: str) -> list:
    """用上次成功的 schema 包一层「MCP 不可用」stub，避免 tools 列表整表消失。"""
    from langchain_core.tools import StructuredTool

    err_text = f"MCP_UNAVAILABLE: 服务「{server_name}」当前不可用：{error[:200]}"
    tools: list = []
    if not isinstance(snaps, list):
        return tools
    for snap in snaps:
        if not isinstance(snap, dict):
            continue
        name = str(snap.get("name") or "").strip()
        if not name:
            continue
        desc = str(snap.get("description") or name)
        raw_schema = snap.get("args_schema")
        schema = _frozen_args_schema(raw_schema if isinstance(raw_schema, dict) else None, tool_name=name)

        async def _coro(*args: Any, _msg: str = err_text, **kwargs: Any) -> str:
            return _msg

        tools.append(
            StructuredTool.from_function(
                coroutine=_coro,
                name=name,
                description=desc,
                args_schema=schema,
            )
        )
    return tools


def _connection_dict(transport: str, url: str, headers: dict[str, str] | None) -> dict[str, Any]:
    conn: dict[str, Any] = {"transport": transport, "url": url}
    if headers:
        conn["headers"] = headers
    return conn


async def _list_tools(transport: str, url: str, headers: dict[str, str] | None) -> list:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from app.utils.egress import build_mcp_httpx_client_factory

    # 连接前再检一次，降低 DNS 重绑定窗口
    conn = _connection_dict(transport, url, headers)
    conn["httpx_client_factory"] = build_mcp_httpx_client_factory(url)
    client = MultiServerMCPClient({"server": conn})
    return await asyncio.wait_for(client.get_tools(), timeout=MCP_CONNECT_TIMEOUT_SECONDS)


async def test_mcp_server_connection(
    transport: str, url: str, headers: dict[str, str] | None
) -> dict[str, Any]:
    """测试连接：成功返回工具名列表；失败抛出异常由调用方包装。"""
    tools = await _list_tools(transport, url, headers)
    return {
        "ok": True,
        "tool_count": len(tools),
        "tool_names": [(getattr(t, "name", "") or "") for t in tools],
    }


def _guard_mcp_tool_output(tool: Any, *, exposed_name: str | None = None) -> Any:
    """包装 MCP 工具：字符串输出按非可信内容隔离并截断，缓解间接提示注入与上下文膨胀。

    :param exposed_name: 对外暴露的名字。`StructuredTool.from_function` 会重建工具，
        这里是唯一能改名的位置——在它之前改原对象、之后又重建，改名就白做了。
    """
    from langchain_core.tools import StructuredTool

    from app.utils.content_guard import guard_untrusted_content

    async def _guarded_coroutine(*args: Any, _tool: Any = tool, **kwargs: Any) -> Any:
        result = await _tool.ainvoke(kwargs if kwargs else (args[0] if args else {}))
        if isinstance(result, str):
            from app.settings import settings

            return guard_untrusted_content(
                result, max_chars=int(getattr(settings, "MCP_TOOL_RESULT_MAX_CHARS", 8000))
            )
        return result

    return StructuredTool.from_function(
        coroutine=_guarded_coroutine,
        name=exposed_name or (getattr(tool, "name", None) or "mcp_tool"),
        description=getattr(tool, "description", "") or "",
        args_schema=getattr(tool, "args_schema", None),
    )


async def load_agent_mcp_tools(agent_id: int) -> tuple[list, list[dict], set[str]]:
    """
    加载指定智能体全部已启用 MCP 服务的 LangChain 工具。
    单个服务连接失败仅记录错误、不中断对话；工具名与内置工具或彼此冲突时跳过。

    :return: `(tools, errors, confirm_names)`
        - errors 元素 `{"name": 服务名, "error": 错误说明}`
        - confirm_names：**需要人工确认的工具名集合**。调用方拿它去填
          `interrupt_on`（U5c）。必须由本函数返回而不是让调用方自己算，因为
          `confirm_policy` 是**按服务行配的**（`UserAgentMcpServer.confirm_policy`），
          工具对象不携带这个信息，调用方无从判定。
    """
    from app.core.cache import cache
    from app.models.user_agent_mcp import UserAgentMcpServer

    rows = await UserAgentMcpServer.filter(agent_id=agent_id, enabled=True).all()
    if not rows:
        return [], [], set()

    all_tools: list = []
    errors: list[dict] = []
    confirm_names: set[str] = set()
    seen_names: set[str] = set(_BUILTIN_TOOL_NAMES) | set(_RESERVED_TOOL_NAMES)

    for row in rows:
        schema_key = mcp_tool_schema_cache_key(agent_id, row.id)
        confirm_policy = getattr(row, "confirm_policy", "auto") or "auto"
        try:
            tools = await _list_tools(row.transport, row.url, decrypt_headers(row.headers_ciphertext))
            cache.set_json(schema_key, _snapshot_mcp_tools(tools), MCP_TOOL_SCHEMA_TTL_SECONDS)
        except Exception as e:
            err_text = str(e)[:300]
            errors.append({"name": row.name, "error": err_text})
            cached = cache.get_json(schema_key)
            if cached:
                logger.warning(
                    "MCP list_tools 失败，使用缓存 schema stub agent_id=%s server=%s: %s",
                    agent_id,
                    row.name,
                    err_text,
                )
                tools = _stub_mcp_tools_from_snapshot(cached, row.name, err_text)
            else:
                continue
        for t in tools:
            raw_name = getattr(t, "name", "") or ""
            if not raw_name:
                errors.append({"name": row.name, "error": "工具名为空，已跳过"})
                continue
            # 命名空间：不同 server 的同名工具从此共存，不再丢一个。
            # 撞名只可能来自「同一 server 内部重名」或「净化/截断后碰撞」，
            # 那种情况加哈希后缀而不是丢弃——丢弃正是这个改动要解决的问题。
            exposed = mcp_exposed_name(row.name, raw_name, server_id=row.id)
            if exposed in seen_names:
                deduped = _dedupe_exposed_name(exposed, seen_names)
                logger.info(
                    "MCP 工具名去重 server=%s raw=%s %s -> %s",
                    row.name,
                    raw_name,
                    exposed,
                    deduped,
                )
                exposed = deduped
            seen_names.add(exposed)
            # 判定必须在 `_guard_mcp_tool_output` **之前**：它用 StructuredTool 重建工具，
            # annotations 会丢，之后再判定就只能看到 name/description（`readOnlyHint`
            # 这类显式标注失效，只读工具会被误判成高危）。
            if should_confirm_mcp_tool(row.name, t, confirm_policy):
                confirm_names.add(exposed)
            all_tools.append(_guard_mcp_tool_output(t, exposed_name=exposed))
    all_tools.sort(key=lambda t: str(getattr(t, "name", "") or ""))
    return all_tools, errors, confirm_names
