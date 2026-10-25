"""对话域 ORM（SQLAlchemy），与主应用 Tortoise 库分离。

为什么移到这里：这些表是**对话域**的（会话/消息/附件/记忆水位线/轮次快照），
但历史上 `app/kb`（知识库）要读同一套 SQLAlchemy 会话与 Base，就把整个
`app.chat.db_models` 拖进了知识库的 import 链。拆分后：
- 对话五表 → `app.core.db_models`
- 知识库三表 → `app.kb.db_models`
两者都只依赖 `app.core.db.Base`（叶子节点），互不 import。

表之间零跨域外键：KB 三表与 Chat 五表之间没有任何 ForeignKey，只靠
`kb_scope` / `session_id` 字符串做逻辑隔离，所以这次拆分不会产生循环依赖。

级联删除、索引、JSON 存储的取舍见各表 docstring。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Float,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class ChatSession(Base):
    """
    聊天会话模型
    id: 主键
    user_id: 用户ID (必填, 有索引)
    agent_id: 智能体ID (必填, 有索引)
    session_id: 会话ID (必填, 有索引)
    metadata_json: 元数据
    updated_at: 更新时间
    created_at: 创建时间
    确保同一用户同一智能体同一会话只存在一个会话
    与 ChatMessage 是一对多关系, 级联删除(当会话删除时，关联的消息也会自动删除)
    """
    __tablename__ = "mg_chat_sessions"
    __table_args__ = (UniqueConstraint("user_id", "agent_id", "session_id", name="uq_mg_user_agent_session"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    agent_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_user_preview: Mapped[str | None] = mapped_column(String(160), nullable=True)
    path_message_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    memory_cursor = relationship(
        "ChatMemoryCursor",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChatMemoryCursor(Base):
    """
    会话记忆归档水位线：已成功写入 Milvus 的最后一轮 turn 索引（从 0 起）。
    与 mg_chat_sessions 一对一，随会话删除级联删除。
    """

    __tablename__ = "mg_chat_memory_cursor"

    session_ref_id: Mapped[int] = mapped_column(
        ForeignKey("mg_chat_sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    last_archived_turn_index: Mapped[int] = mapped_column(Integer, nullable=False, default=-1)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    session: Mapped[ChatSession] = relationship("ChatSession", back_populates="memory_cursor")


class ChatMessage(Base):
    """
    聊天消息模型
    id: 主键
    session_ref_id: 会话ID (外键, 关联到 mg_chat_sessions.id, 级联删除)
    parent_id: 父消息ID (自引用, 构成会话内消息树; 根消息为空)
    selected_child_id: 当前选中的子消息ID (沿它解析当前分支路径; 叶子为空)
    message_type: 消息类型
    content: 消息内容 (文本)
    content_json: LangChain 消息整块 JSON（含多模态 image_ref 等），优先于 content 回放
    timestamp: 消息时间 (默认值为当前时间)
    rag_trace: 检索增强生成跟踪 (可选)
    rag_steps: 流式检索步骤列表 (可选，用于历史回放)
    error_text: 本条助手消息生成失败时的错误文案 (可选)
    thinking_text: 工具调用前的过渡文本 (可选，用于历史回放思考区)
    approval_pending: 审批中断时的待确认信息 (可选，有值时代表等待用户决定)
    通过 session 属性回关到 ChatSession
    """
    __tablename__ = "mg_chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_ref_id: Mapped[int] = mapped_column(
        ForeignKey("mg_chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("mg_chat_messages.id", ondelete="CASCADE"), nullable=True, index=True
    )
    selected_child_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    message_type: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    rag_trace: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    rag_steps: Mapped[list | None] = mapped_column(JSON, nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_references: Mapped[list | None] = mapped_column(JSON, nullable=True)  # 图片引用列表
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)  # 知识库来源列表（与回答中 [来源N] 编号对应）
    thinking_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # 工具调用前的过渡文本（思考区内容）
    thinking_items: Mapped[list | None] = mapped_column(JSON, nullable=True)  # 思考区有序时间线：step/text 交错
    # 审批中断时的待确认信息（interrupt / actions / thread_id）。有值时表示这条助手消息是
    # 「已暂停等用户决定」的中间态，resume 到手后会被原地覆盖为最终回答。
    approval_pending: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    session = relationship("ChatSession", back_populates="messages")


class ChatAttachment(Base):
    """
    会话内用户上传附件（与知识库 mg_kb_documents 分离）。
    id: 主键
    user_id: 用户ID (必填, 有索引)
    agent_id: 智能体ID (必填, 有索引)
    session_id: 会话ID (必填, 有索引)
    original_filename: 原始文件名
    stored_relpath: 存储路径
    mime: 文件类型
    size_bytes: 文件大小
    kind: 文件类型
    created_at: 创建时间
    """

    __tablename__ = "mg_chat_attachments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    agent_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    stored_relpath: Mapped[str] = mapped_column(String(1024), nullable=False)
    mime: Mapped[str] = mapped_column(String(128), nullable=False, default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="other")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ChatTurnEvent(Base):
    """一轮对话的「内容状态快照」，用于 Redis 事件流丢失后的恢复。

    为什么不逐条存事件：一轮对话的 content 事件是 token 级的（单轮 300–1500 条），
    逐条 INSERT 会把流式延迟吃掉；而前端 `since_seq` 是本地自增计数器（数自己收到几条），
    seq 一旦出现空洞，重连时计数器就会与服务端序号错位。
    折中做法：只存「某时刻的累积文本 + 该时刻的事件序号」，断线时整体恢复到最后一次快照。

    kind:
    - `snapshot`：生成中途的周期快照
    - `done` / `error` / `cancelled`：终态；写入时会删除该 job 的所有 snapshot，
      保证每个 job 最终只留 1 行，表增长被压到「每轮对话 1 行」。

    易错点：不要给 (job_id, seq) 建唯一约束——同一个 seq 可能先后写入 snapshot 与 done
    两种 kind，去重语义由「终态时清 snapshot」保证，读时取 seq 最大的一条。
    """

    __tablename__ = "mg_chat_turn_events"
    __table_args__ = (
        Index("ix_mg_turn_events_job", "job_id", "seq"),
        Index("ix_mg_turn_events_created", "created_at"),
    )

    # PostgreSQL 用 BIGSERIAL；sqlite（单测）只对精确的 INTEGER PRIMARY KEY 做自增，
    # 直接用 BigInteger 会导致 NOT NULL constraint failed: id
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    job_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_id: Mapped[int] = mapped_column(Integer, nullable=False)
    session_id: Mapped[str] = mapped_column(String(120), nullable=False)
    # 快照覆盖到的最后一个事件序号；断线重连时用它告诉前端「已生成到这里」
    seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="snapshot")
    # 用通用 JSON 而非 JSONB：payload 从不参与查询/索引，且这样单测可用 sqlite 建表
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ChatMemoryItem(Base):
    """**跨会话**的长期记忆条目（U7）。与会话级记忆是两套东西，别混。

    区别：
    - 会话级（`chiron_ai_chat_memory` 集合，scope `u{u}_a{a}_s{sid}`）存的是**原话切片**，
      只在轮次被 compact 挤出原文窗口时归档，用来补"本会话早期说过什么"。
    - 本表存的是**提炼过的事实**，跨会话可见，用来补"这个用户/这个智能体长期怎么做事"。

    `mem_key` + **部分唯一索引**是整张表的核心设计：**按主题 upsert，不是无脑追加**。
    没有它，三个月前"我用 MySQL"和现在"我换 PostgreSQL 了"会同时留在库里，
    检索时随机命中一条。长期记忆必须是「可被改写的事实」，不是「只增不减的流水账」。

    易错点：这里**不能**用普通 `UniqueConstraint`。改写时要保留旧值可追溯
    （旧行标 `status='superseded'`、新行 `active`），普通唯一约束会让两条同 key 的行
    无法共存。所以用 `WHERE status='active'` 的**部分唯一索引**——既能保证"同时只有
    一条生效"，又允许历史版本留存。

    `embedding_synced=false` 的记录由后台补扫同步到 Milvus——双写不追求强一致，
    PG 是真相源，Milvus 落后最多导致"召回少一条"。
    """

    __tablename__ = "mg_chat_memories"
    __table_args__ = (
        # 只约束「生效中的那条」唯一，历史版本（superseded）可以多条共存
        Index(
            "uq_mem_active",
            "user_id",
            "agent_id",
            "mem_key",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
        Index("ix_mem_user_agent_status", "user_id", "agent_id", "status"),
        Index("ix_mem_embedding_synced", "embedding_synced"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_id: Mapped[int] = mapped_column(Integer, nullable=False)
    # 主题键：稳定标识「这条记忆说的是哪件事」，同类事实靠它覆盖
    mem_key: Mapped[str] = mapped_column(String(120), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # fact / preference / convention / decision
    kind: Mapped[str] = mapped_column(String(24), nullable=False, default="fact")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    # 来源可追溯：用户要能问「你凭什么记这条」
    source_session: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_turn_key: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # active / superseded / deleted
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    embedding_synced: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
