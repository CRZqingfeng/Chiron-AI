from typing import Optional

from app.core.crud import CRUDBase
from app.models.user_agent import UserAgent
from app.schemas.user_agent import UserAgentCreate, UserAgentUpdate
from app.services.user_agent_access import get_accessible_agent


class UserAgentController(CRUDBase[UserAgent, UserAgentCreate, UserAgentUpdate]):
    def __init__(self):
        super().__init__(model=UserAgent)

    async def get_owned(self, agent_id: int, user_id: int) -> Optional[UserAgent]:
        return await self.model.filter(id=agent_id, user_id=user_id).first()

    async def get_accessible(self, agent_id: int, user_id: int) -> Optional[UserAgent]:
        """属主本人，或已发布且当前用户在共享名单内。

        真实实现已下沉到 `app.services.user_agent_access`：对话执行链路也要用它，
        而 `app.chat` 不能依赖 `app.controllers`。这里只做门面转发，
        API 层的 8 处调用保持原样。
        """
        return await get_accessible_agent(agent_id, user_id)


user_agent_controller = UserAgentController()
