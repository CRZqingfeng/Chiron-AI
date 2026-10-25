"""知识库域 ORM（SQLAlchemy）。

从 `app.chat.db_models` 拆出：这三张表只服务知识库，却让整个 `app.kb` 的 import
链必须依赖 `app.chat`。拆开后 `app.kb` 只依赖 `app.core.db.Base`。

KB 三表与 Chat 五表之间**零外键**：只靠 `kb_scope` 字符串做逻辑隔离，
所以本模块与 `app.core.db_models` 之间没有循环依赖。

嵌入向量不在这里——它们在 Milvus；这里只存元数据与父块原文（Auto-merge 用）。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class KbDocument(Base):
    """
    智能体知识库：已上传文件元数据（展示名为原始上传文件名）。
    :param id: 主键
    :param kb_scope: 知识库范围
    :param display_filename: 展示文件名
    :param stored_filename: 存储文件名
    :param file_type: 文件类型
    :param chunk_count: 分块数量
    :param created_at: 创建时间
    :param updated_at: 更新时间
    """

    __tablename__ = "mg_kb_documents"
    __table_args__ = (UniqueConstraint("kb_scope", "display_filename", name="uq_mg_kb_scope_display_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    kb_scope: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    display_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 文件内容 sha256，用于同内容重传跳过重建
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class KbParentChunk(Base):
    """
    知识库父级分块（L1/L2），用于 Auto-merge。
    :param chunk_id: 分块ID
    :param kb_scope: 知识库范围
    :param parent_chunk_id: 父级分块ID
    :param root_chunk_id: 根级分块ID
    :param chunk_level: 分块层级
    :param chunk_idx: 分块索引
    :param updated_at: 更新时间
    """

    __tablename__ = "mg_kb_parent_chunks"

    chunk_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    kb_scope: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    filename: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    file_type: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    parent_chunk_id: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    root_chunk_id: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    chunk_level: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_idx: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class KbImage(Base):
    """
    知识库图片元数据表，用于存储从文档中提取的图片信息。
    :param id: 图片唯一ID
    :param kb_scope: 知识库范围
    :param filename: 原始文件名
    :param display_filename: 展示文件名
    :param stored_relpath: 存储相对路径
    :param file_size: 文件大小（字节）
    :param mime_type: MIME类型
    :param width: 图片宽度
    :param height: 图片高度
    :param format: 图片格式
    :param caption: 预留字段（可选说明文字，当前不入库生成）
    :param embedding_model: 使用的嵌入模型
    :param source_document: 来源文档名
    :param page_number: 所在页码
    :param position_in_doc: 在文档中的位置信息
    :param chunk_id: 关联的向量块ID
    :param parent_chunk_id: 关联的父块ID
    :param root_chunk_id: 关联的根块ID
    :param created_at: 创建时间
    :param updated_at: 更新时间
    """

    __tablename__ = "mg_kb_images"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kb_scope: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    
    # 文件信息
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    display_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    stored_relpath: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=True)
    
    # 图片属性
    width: Mapped[int] = mapped_column(Integer, nullable=True)
    height: Mapped[int] = mapped_column(Integer, nullable=True)
    format: Mapped[str] = mapped_column(String(20), nullable=True)
    
    # 语义信息
    caption: Mapped[str] = mapped_column(Text, nullable=True, default="")  # 预留，默认空
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    
    # 关联信息
    source_document: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # 图片在页面中的位置信息
    position_x: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    position_y: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    position_width: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    position_height: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    
    # 检索相关
    chunk_id: Mapped[str] = mapped_column(String(512), nullable=True, index=True)
    parent_chunk_id: Mapped[str] = mapped_column(String(512), nullable=True, index=True)
    root_chunk_id: Mapped[str] = mapped_column(String(512), nullable=True, index=True)
    
    # 图文关联
    related_text_ids: Mapped[list] = mapped_column(JSONB, nullable=True, default=list)  # 关联的文本块ID列表
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
