"""知识库工具 provider（search_knowledge_base / search_knowledge_by_image）。

两个纪律片段来自 `agent_prompt._KB_IMAGE_DISCIPLINE` / `_KB_ANSWER_DISCIPLINE`，
搬到这里与工具定义同居。
"""

from __future__ import annotations

from typing import Any

from app.chat.tool_providers.base import ToolBuildContext

_KB_IMAGE_DISCIPLINE = (
    "知识库检索结果中，每个图片 chunk 都会单独给出一行现成的 Markdown：`![说明](/api/v1/media/...?exp=...&sig=...)`。"
    "展示图片时必须把那一行原样复制到回答中，括号内必须是以 /api/v1/media/ 开头并带 ?exp=&sig= 的地址。"
    "禁止自行改写或拼接括号内内容：不得填入文档名、页码、`[i] ... (Page n)` 标题、stored_relpath、"
    "不得改成 http(s) 绝对地址、image://、file://、kb_image://，也不得用 [1][2] 或序号代替。"
)

_KB_ANSWER_DISCIPLINE = (
    "知识库作答纪律：回答必须仅依据知识库检索工具的返回内容与多轮对话上下文；"
    "凡引用检索到的内容，必须以 [来源N] 标注（N 与工具返回中的编号一致）。"
    "当工具返回明确提示知识库无相关资料（或检索未命中）或本轮禁用（TOOL_DISABLED_THIS_TURN）时，"
    "必须如实告知用户「知识库中未找到相关资料」"
    "并说明可补充资料后重试，不得编造知识库结论或凭想象作答；"
    "若无确凿资料支撑，宁可说明「知识库中未找到相关资料」，也不要虚构。"
    "注意区分知识库中的结论与你的一般常识推断，后者不得冒充知识库内容。"
)


class KbToolProvider:
    """知识库检索（含以图检索）。"""

    name = "kb"
    order = 30

    def enabled(self, ctx: ToolBuildContext) -> bool:
        """工具始终注册——本轮禁用由 `tools.py` 的 turn policy 在调用时返回提示。"""
        return True

    def build(self, ctx: ToolBuildContext) -> list[Any]:
        from app.kb.image_search_tool import make_search_knowledge_by_image_tool
        from app.kb.search_tool import make_search_knowledge_tool

        return [
            make_search_knowledge_tool(
                ctx.kb_scope,
                ctx.llm_config,
                knowledge_base_document_filter=ctx.knowledge_base_document_filter,
            ),
            make_search_knowledge_by_image_tool(
                ctx.kb_scope, ctx.user_id, ctx.agent_id, ctx.session_id
            ),
        ]

    def prompt_fragments(self, ctx: ToolBuildContext) -> list[str]:
        """本轮不用知识库时不给纪律——否则模型会为"不能用的工具"自我约束。"""
        if not ctx.use_knowledge_retrieval:
            return []
        return [_KB_IMAGE_DISCIPLINE, _KB_ANSWER_DISCIPLINE]


PROVIDER = KbToolProvider()
