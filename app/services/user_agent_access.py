"""智能体可访问性判定：属主本人，或已发布且当前用户在共享名单内。

位置说明：原先住在 `app/controllers/user_agent.py` 的控制器方法里。但它是
**业务规则 + 纯数据查询**，不涉及 HTTP；而对话执行链路（`app.chat.chat_job`）
也要用它来判断某个 job 有没有权限跑这个智能体。留在 controllers 里会让
`app.chat` 反向依赖 `app.controllers`（分层反向）。

`UserAgentController.get_accessible` 保留为门面方法（API 层 8 处调用不变），
内部转调本函数 —— 门面转发，不是逻辑复制。
"""

from __future__ import annotations

from typing import Optional

from tortoise.expressions import Q

from app.models.user_agent import UserAgent


async def get_accessible_agent(agent_id: int, user_id: int) -> Optional[UserAgent]:
    """属主本人，或已发布且当前用户在共享名单内；否则返回 None。"""
    return (
        await UserAgent.filter(id=agent_id)
        .filter(Q(user_id=user_id) | Q(is_published=True, shares__user_id=user_id))
        .first()
    )
