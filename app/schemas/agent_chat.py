from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ChatRequest(BaseModel):
    """
    智能体对话请求
    :param agent_id: 智能体 ID
    :param message: 用户消息
    :param session_id: 会话 ID，前端生成
    :param use_knowledge_retrieval: 为 True 时允许知识库检索工具；为 False 时仅通用知识回答
    :param use_web_search: 为 True 时允许联网搜索工具（与知识库检索互斥，不可同时开启）
    :param attachment_ids: 本会话已上传附件 ID 列表
    :param regenerate: 为 True 时重新生成最后一轮助手回复（不新增用户消息；附件以存储中的最后一条用户消息为准）
    """
    agent_id: int = Field(..., description="智能体 ID")
    message: str = Field("", description="用户消息；可与 attachment_ids 二选一或同时使用")
    session_id: Optional[str] = Field("default_session", description="会话 ID，前端生成")
    use_knowledge_retrieval: bool = Field(
        True,
        description="为 True 时允许知识库检索工具；为 False 时仅通用知识回答",
    )
    use_web_search: bool = Field(
        False,
        description="为 True 时允许联网搜索工具（DuckDuckGo）；与知识库检索互斥",
    )
    attachment_ids: list[str] = Field(default_factory=list, description="本会话已上传附件 ID 列表")
    regenerate: bool = Field(
        False,
        description="为 True 时重新生成助手回复（不新增用户消息；附件以存储中的对应用户消息为准）",
    )
    target_message_id: Optional[int] = Field(
        None,
        description="regenerate 时要重写的助手消息 ID；缺省时回退为当前路径末尾的助手消息",
    )
    mcp_approved_pending_id: Optional[str] = Field(
        None,
        deprecated=True,
        description=(
            "【已废弃，U5c】MCP 高危工具已统一走审批中断（resume_decision / resume_thread_id）。"
            "保留该字段只为不把旧客户端的请求打成 422；后端不解它。"
        ),
    )
    resume_thread_id: Optional[str] = Field(
        None,
        description="审批恢复：中断那一轮的 job_id（checkpoint 线程标识）。与 resume_decision 成对使用",
    )
    resume_decision: Optional[str] = Field(
        None,
        description="审批决定：approve / reject。给定时从断点继续，不追加新的用户消息",
    )
    @field_validator("message")
    @classmethod
    def _limit_message_length(cls, v: str) -> str:
        from app.settings import settings

        mx = int(getattr(settings, "CHAT_MESSAGE_MAX_LENGTH", 8000))
        if v and len(v) > mx:
            raise ValueError(f"消息长度不能超过 {mx} 字符")
        return v

    @field_validator("resume_decision")
    @classmethod
    def _check_resume_decision(cls, v: Optional[str]) -> Optional[str]:
        """只接受 approve / reject。

        不在入口拦住的话，非法值会一路传到 LangGraph 的 HITL middleware，
        在那抛 ValueError 后被错误分类器归为 unknown，用户只看到「生成失败」——
        既查不出原因，也没有可操作的提示。
        """
        if v is None or not str(v).strip():
            return None
        val = str(v).strip().lower()
        if val not in ("approve", "reject"):
            raise ValueError("resume_decision 只能是 approve 或 reject")
        return val

    @field_validator("session_id")
    @classmethod
    def _limit_session_id_length(cls, v: Optional[str]) -> Optional[str]:
        from app.settings import settings

        if v is None:
            return v
        mx = int(getattr(settings, "CHAT_SESSION_ID_MAX_LENGTH", 64))
        if len(v.strip()) > mx:
            raise ValueError(f"会话 ID 长度不能超过 {mx} 字符")
        return v

    @field_validator("attachment_ids")
    @classmethod
    def _limit_attachment_ids(cls, v: list[str]) -> list[str]:
        from app.settings import settings

        mx = int(getattr(settings, "CHAT_UPLOAD_MAX_FILES_PER_MESSAGE", 5))
        if len(v) > mx:
            raise ValueError(f"单条消息引用附件数量不能超过 {mx}")
        return v

    @model_validator(mode="after")
    def _message_or_attachments(self) -> "ChatRequest":
        if self.regenerate:
            return self
        # 审批恢复既没有新消息、也没有附件：模型输入完全来自 checkpoint，
        # `message` 本来就该是空的。不加这一条，恢复请求会在入口被 422 拦掉。
        if self.resume_decision:
            return self
        if not (self.message or "").strip() and not self.attachment_ids:
            raise ValueError("message 与 attachment_ids 至少填写一项")
        return self

    @model_validator(mode="after")
    def _retrieval_switches_exclusive(self) -> "ChatRequest":
        if self.use_knowledge_retrieval and self.use_web_search:
            raise ValueError("知识库检索与联网搜索不能同时开启，请只选择其一")
        return self


class ChatAttachmentUploadResponse(BaseModel):
    """
    会话附件上传结果
    :param id: 附件 ID
    :param filename: 附件文件名
    :param kind: 附件类型
    :param mime: 附件 MIME 类型
    :param size_bytes: 附件大小（字节）
    """

    id: str
    filename: str
    kind: str
    mime: str
    size_bytes: int


class BranchSelectRequest(BaseModel):
    """切换助手回复版本：把目标助手消息设为其父用户消息下的选中分支。"""

    assistant_message_id: int = Field(..., description="要切换到的助手消息 ID")


class MessageInfo(BaseModel):
    """
    消息信息
    :param message_id: 消息行 ID（消息树中的稳定身份）
    :param type: 消息类型
    :param content: 消息内容（文本预览）
    :param content_json: LangChain 消息整块 JSON（多模态 image_ref 等）
    :param timestamp: 消息时间戳
    :param rag_trace: RAG 追踪信息
    :param rag_steps: 检索步骤（与 SSE rag_step 一致，用于历史回放）
    :param error_text: 助手消息生成失败时的错误说明（可选，用于历史展示）
    :param thinking_text: 工具调用前的过渡文本（可选，用于历史回放思考区）
    :param thinking_items: 思考区有序时间线（step/text 交错，可选，用于历史回放）
    :param approval_pending: 审批中断的待确认信息（可选，有值时表示这条消息在等用户决定）
    :param version_index: 助手消息在同父兄弟版本中的序号（从 1 起）
    :param version_count: 同父兄弟版本总数
    :param sibling_ids: 同父兄弟版本的消息 ID（按版本序号排序，含自身）
    """
    message_id: Optional[int] = None
    type: str
    content: Any = ""
    content_json: Optional[dict[str, Any]] = None
    timestamp: str
    rag_trace: Optional[dict[str, Any]] = None
    rag_steps: Optional[list[dict[str, Any]]] = None
    error_text: Optional[str] = None
    sources: Optional[list[dict[str, Any]]] = None
    thinking_text: Optional[str] = None
    thinking_items: Optional[list[dict[str, Any]]] = None
    # 为什么必须显式声明：pydantic 默认丢弃未声明字段，storage 里存了也传不出去。
    # 少了这一项，中断后刷新页面确认卡片会消失——用户再也点不到「允许」。
    approval_pending: Optional[dict[str, Any]] = None
    version_index: int = 1
    version_count: int = 1
    sibling_ids: Optional[list[int]] = None


class SessionMessagesResponse(BaseModel):
    """
    会话消息响应
    :param messages: 消息列表
    """
    messages: list[MessageInfo]


class SessionInfo(BaseModel):
    """
    会话信息
    :param session_id: 会话 ID
    :param updated_at: 更新时间
    :param message_count: 消息数量
    :param last_user_preview: 最后用户预览
    :param agent_id: 智能体 ID
    :param agent_name: 智能体名称
    :param updated_at_display: 更新时间展示
    """
    session_id: str
    updated_at: str
    message_count: int
    last_user_preview: str = ""
    agent_id: int | None = None
    agent_name: str = ""
    updated_at_display: str = ""


class SessionListResponse(BaseModel):
    """
    会话列表响应
    :param sessions: 会话列表
    :param total: 总数量
    :param has_more: 是否有更多
    """
    sessions: list[SessionInfo]
    total: int = 0
    has_more: bool = False


class SessionDeleteResponse(BaseModel):
    """
    会话删除响应
    :param session_id: 会话 ID
    :param message: 消息
    """
    session_id: str
    message: str


class ChatJobCreateResponse(BaseModel):
    """创建异步对话 Job 的响应（刷新后可按 job_id 重连 SSE）。"""

    job_id: str
