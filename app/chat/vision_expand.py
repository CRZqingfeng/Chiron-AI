"""把消息里的 `image_ref` 展开成模型能看的格式（需要读会话附件）。

位置说明：这两个函数原先住在 `app/core/message_codec.py`，但它们要读附件表/对象存储
（`app.chat.attachment_service`），放在 core 会让 core 反向依赖 chat。
纯编解码（JSON 信封、占位替换、图片字节预处理）仍在 core，这里只管「取字节」。

注意 `expand_*` 是**同步**函数：附件读取走本地文件/对象存储的同步客户端。
调用方（`agent_service`）在 async 上下文里通过 `asyncio.to_thread` 或流式前置阶段使用。
"""

from __future__ import annotations

import base64
import copy
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from app.core.message_codec import prepare_vision_image_bytes


def expand_human_image_refs(
    content: Any,
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    expand_images: bool = True,
) -> Any:
    """
    对 image_ref：默认读盘后编成 OpenAI 兼容 data URL。
    expand_images=False 时改为短占位，避免历史图撑满上下文。
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    # 延迟导入：1) 避免本模块被 import 时就拖起 storage/db 依赖；
    # 2) 测试里 mock 的是 `app.chat.attachment_service.file_bytes_for_attachment` 这个目标，
    #    顶层 from-import 会把名字绑死在这里，mock 就失效了。
    from app.chat.attachment_service import file_bytes_for_attachment

    new_blocks: list[Any] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "file_ref":
            continue
        if isinstance(block, dict) and block.get("type") == "image_ref":
            aid = str(block.get("attachment_id") or "").strip()
            if not aid:
                continue
            fn = (block.get("filename") or "").strip()
            if not expand_images:
                label = f"[图片 attachment_id={aid}"
                if fn:
                    label += f" filename={fn}"
                label += "]"
                new_blocks.append({"type": "text", "text": label})
                continue
            raw = file_bytes_for_attachment(
                aid, user_id=user_id, agent_id=agent_id, session_id=session_id
            )
            if not raw:
                new_blocks.append({"type": "text", "text": f"[图片附件 {aid} 已不可用]"})
                continue
            mime = block.get("mime") or "image/png"
            payload, out_mime = prepare_vision_image_bytes(raw, mime=str(mime))
            b64 = base64.b64encode(payload).decode("ascii")
            image_url: dict[str, Any] = {"url": f"data:{out_mime};base64,{b64}"}
            if fn:
                image_url["filename"] = fn
            new_blocks.append({"type": "image_url", "image_url": image_url, "filename": fn})
        else:
            new_blocks.append(copy.deepcopy(block))
    return new_blocks


def expand_messages_for_model(
    messages: list[BaseMessage],
    *,
    user_id: int,
    agent_id: int,
    session_id: str,
    images_on_last_human_only: bool = False,
    expand_images: bool = True,
) -> list:
    """
    把 messages 里的图片引用（image_ref）转成模型能看的格式
    要么展开成 base64 data URL，要么转成占位文本（两阶段读图时用）
    images_on_last_human_only=True 时仅最后一条 Human 展开为 data URL。
    expand_images=False 时全部 image_ref 变占位文本（两阶段读图：agent 阶段不见图）。
    """
    last_human_i = None
    if images_on_last_human_only:
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_i = i
                break
    out: list = []
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            expand = expand_images
            if expand and images_on_last_human_only:
                expand = i == last_human_i
            expanded = expand_human_image_refs(
                m.content,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                expand_images=expand,
            )
            out.append(HumanMessage(content=expanded))
        else:
            out.append(m)
    return out
