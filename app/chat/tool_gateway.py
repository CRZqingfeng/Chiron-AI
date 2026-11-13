"""工具可见性网关：MCP 工具太多时，不把它们全塞进模型上下文。

## 为什么需要

实测（U11 方案 §1）：单个 MCP 工具定义平均 **331 tokens**。100 个工具 = 19.3k tokens
（占 128k 上下文的 15%），400 个 = 60%，900 个直接超出上下文发不出请求。

比 token 更糟的是准确率：公开实测里 100+ 工具时 tool-call 准确率约 13–15%，
某模型从 207 涨到 417 个工具时从 64% 掉到 20%（接近随机）。

## 做法：不改注册，只改「每轮让模型看到什么」

工具**全集仍然注册给图**（`create_agent(tools=...)` 不变），tool node 认得所有工具；
本中间件只在 `wrap_model_call` 里裁剪 `request.tools`，模型这轮只看到：

    core（内置工具 + search_tools）  ∪  本轮被搜索激活的长尾工具

好处是**不需要重建 agent**：激活后工具直接可调用，因为 tool node 早就认识它。

## 为什么默认不启用

小工具集下全量注入准确率最高、少一次 LLM 往返。为了架构统一而牺牲小场景，
是过度设计。所以默认 `auto`：只有 MCP 工具数超过阈值才切按需发现，
**未超阈值时模型看到的工具集合与改动前逐字相同**（有测试守着）。

## 与审批的关系（容易被漏）

`interrupt_on` 按工具名配置，裁剪可见性**不影响**它——工具仍在图里，
模型调它时 HITL 照常中断。反过来说：**可见性不是安全边界**，
只是上下文优化。真正的安全边界在 `app/chat/approval.py`。
"""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

from app.core.turn_runtime import activate_tools, get_activated_tools
from app.log import logger
from app.settings import settings

SEARCH_TOOL_NAME = "search_tools"
"""常驻的元工具名。它自己永远可见，否则模型搜过一次之后就再也搜不了。"""

_DEFAULT_SEARCH_LIMIT = 8

# 打分权重：名字命中比描述命中更有判别力。
# 一个工具的描述常写得很长（"这是一个用于…的工具"），词面重合是噪音。
_NAME_HIT_WEIGHT = 3.0
_DESC_HIT_WEIGHT = 1.0


def _gateway_mode() -> str:
    return str(getattr(settings, "TOOL_GATEWAY_MODE", "auto") or "auto").strip().lower()


def gateway_active(mcp_tool_count: int) -> bool:
    """本轮是否启用按需发现。

    | 模式 | 行为 |
    |---|---|
    | `always_full` | 永远不启用（= 改动前的行为） |
    | `always_discovery` | 永远启用（调试用） |
    | `auto`（默认） | MCP 工具数超过阈值才启用 |

    **阈值只看 MCP 工具数量，不看总数**：内置工具是"手脚"，每轮都要用，
    永远全量。把它们算进阈值会让内置工具一多就误触发。
    """
    mode = _gateway_mode()
    if mode == "always_full":
        return False
    if mode == "always_discovery":
        return True
    threshold = int(getattr(settings, "TOOL_GATEWAY_MAX_MCP_TOOLS", 25) or 25)
    return mcp_tool_count > threshold


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "")


def _tool_description(tool: Any) -> str:
    return str(getattr(tool, "description", "") or "")


def _server_of(name: str) -> str:
    """从 `{server}__{tool}` 反解服务名；没有前缀时返回空串。

    依赖 `app/mcp_client.service.mcp_exposed_name` 定的分隔符，
    所以 MCP 工具名一旦改了拼法，这里要一起改。
    """
    if "__" not in name:
        return ""
    return name.rsplit("__", 1)[0]


def _tokens(text: str) -> list[str]:
    """切成用于打分的词。

    中文没有空格，按字符 bigram 切：`创建` 之类能命中，又不会因为整句匹配不上而归零。
    英文按词切。
    """
    text = (text or "").lower()
    words = re.findall(r"[a-z0-9_]+", text)
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", text)
    grams = [cjk[i : i + 2] for i in range(max(0, len(cjk) - 1))]
    return words + grams


def score_candidate(query: str, name: str, description: str) -> float:
    """候选工具的相关性打分。P0-B 是词面匹配，P0-C 换成 BM25 + 向量混合。"""
    q = _tokens(query)
    if not q:
        return 0.0
    name_l = (name or "").lower()
    desc_l = (description or "").lower()
    score = 0.0
    for token in set(q):
        if token in name_l:
            score += _NAME_HIT_WEIGHT
        if token in desc_l:
            score += _DESC_HIT_WEIGHT
    # 完整 query 直接出现在工具名里，几乎确定就是它
    if len(query.strip()) >= 2 and query.strip().lower() in name_l:
        score += 5.0
    return score


class VisibleToolsMiddleware(AgentMiddleware):
    """把模型的可见工具裁到「常驻 + 本轮激活」。

    `core_names` 由调用方算好传入（= 全量工具名 − MCP 工具名 + `search_tools`）。
    这样这个类不需要知道哪些是 MCP、哪些是内置——判断只在一处发生。
    """

    def __init__(self, core_names: set[str]) -> None:
        super().__init__()
        self.core_names: frozenset[str] = frozenset(core_names)

    def _visible(self, tools: list[Any]) -> list[Any]:
        allowed = set(self.core_names) | get_activated_tools()
        return [t for t in tools if _tool_name(t) in allowed]

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        try:
            tools = self._visible(list(request.tools))
            return handler(request.override(tools=tools))
        except Exception as e:  # noqa: BLE001
            # 裁剪失败必须退回全量：少给几个工具是体验问题，给不出工具是功能故障
            logger.warning("工具可见性裁剪失败，本轮退回全量：{}", e)
            return handler(request)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        try:
            tools = self._visible(list(request.tools))
            return await handler(request.override(tools=tools))
        except Exception as e:  # noqa: BLE001
            logger.warning("工具可见性裁剪失败，本轮退回全量：{}", e)
            return await handler(request)


def make_search_tools_tool(candidates: list[Any]) -> Any:
    """构造 `search_tools` 元工具。

    返回的是 L1 摘要（名字 + 一句话），**完整参数 schema 不给**——
    激活后下一轮模型会在工具列表里看到完整定义，不需要在这里重复一遍
    （那个 schema 就是我们要省的 token）。
    """
    from langchain_core.tools import StructuredTool

    limit_default = int(getattr(settings, "TOOL_GATEWAY_SEARCH_LIMIT", _DEFAULT_SEARCH_LIMIT) or 8)

    catalog = [
        {
            "name": _tool_name(t),
            "description": _tool_description(t),
            "server": _server_of(_tool_name(t)),
        }
        for t in candidates
        if _tool_name(t)
    ]

    def _one_line(text: str, limit: int = 90) -> str:
        text = " ".join((text or "").split())
        return text if len(text) <= limit else text[:limit] + "…"

    async def _search(query: str, limit: int = limit_default) -> str:
        q = (query or "").strip()
        if not q:
            return "错误：query 不能为空。请用一句话说明你想做什么。"

        scored = [
            (score_candidate(q, c["name"], c["description"]), c) for c in catalog
        ]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: (-x[0], x[1]["name"]))
        n = max(1, min(int(limit or limit_default), limit_default))
        top = [c for _, c in scored[:n]]

        if not top:
            # 空结果 = 模型断定「没有这个能力」然后开始编。所以给近似项：
            # 宁可给几个不对的，也让模型知道「工具是有的，只是没匹配上」。
            fallback = sorted(catalog, key=lambda c: c["name"])[:3]
            if not fallback:
                return "当前没有可搜索的外部工具。请直接用已有工具或常识回答。"
            lines = [
                f"未找到与「{q}」匹配的工具。你已知的工具里最接近的是：",
                *[f"- {c['name']}：{_one_line(c['description'])}" for c in fallback],
                "如果都不对，请把你想做的事说得更具体一些再搜一次。",
            ]
            return "\n".join(lines)

        activate_tools([c["name"] for c in top])
        lines = [f"已找到 {len(top)} 个工具，它们已加入你本轮的可用工具列表："]
        for c in top:
            server = f"（{c['server']}）" if c["server"] else ""
            lines.append(f"- {c['name']}{server}：{_one_line(c['description'])}")
        lines.append("现在可以直接调用它们，参数以工具定义为准。")
        return "\n".join(lines)

    return StructuredTool.from_function(
        coroutine=_search,
        name=SEARCH_TOOL_NAME,
        description=(
            "搜索可用的外部工具（MCP 等）。当前接入的工具较多，没有全部列在你的工具列表里。\n"
            "何时使用：需要的能力在现有工具里找不到时；或不确定该用哪个工具时。\n"
            "query 用一句话说明你想做的事（如「创建 GitHub issue」「查 Jira 工单」）。\n"
            "调用后匹配到的工具会出现在你的可用工具列表里，下一轮即可直接调用。"
        ),
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "一句话描述你要做的事"},
                "limit": {"type": "integer", "description": "最多返回几个，默认 8"},
            },
            "required": ["query"],
        },
    )


GATEWAY_PROMPT_FRAGMENT = (
    "## 外部工具（按需发现）\n"
    "你接入的外部工具数量较多，为避免干扰，**没有全部列在你的工具列表里**。\n"
    "当前只列出常用工具；需要其它能力时，先调用 `search_tools`，"
    "用一句话说明要做的事，匹配到的工具会加入你的可用工具列表，下一轮直接调用即可。\n"
    "不要因为「工具列表里没有」就断定做不到——先搜一次再判断。"
)
