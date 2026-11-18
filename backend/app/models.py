from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func

from sqlalchemy import Boolean, Float

from app.database import Base


class User(Base):
    """用户表模型，用于后续登录、问答记录归属等功能。"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, comment="用户ID")
    username = Column(String(50), unique=True, nullable=False, index=True, comment="用户名")
    password_hash = Column(String(255), nullable=False, comment="加密后的密码")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")


class HistoryKnowledge(Base):
    """历史知识表模型，用于存储结构化历史知识点。"""

    __tablename__ = "history_knowledge"

    id = Column(Integer, primary_key=True, index=True, comment="知识点ID")
    title = Column(String(100), nullable=False, unique=True, index=True, comment="知识点标题")
    dynasty = Column(String(50), nullable=True, index=True, comment="所属朝代或时期")
    time = Column(String(100), nullable=True, comment="历史时间")
    people = Column(Text, nullable=True, comment="相关人物，使用 JSON 字符串存储")
    event = Column(Text, nullable=True, comment="历史事件")
    reason = Column(Text, nullable=True, comment="事件原因")
    impact = Column(Text, nullable=True, comment="历史影响")
    keywords = Column(Text, nullable=True, comment="关键词，使用 JSON 字符串存储")
    content = Column(Text, nullable=False, comment="知识点正文内容")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间",
    )


class SignedTextbookKnowledge(Base):
    """Signed lesson 5-20 points; kept separate from the legacy title-keyed table."""

    __tablename__ = "signed_textbook_knowledge"

    knowledge_id = Column(String(64), primary_key=True)
    lesson_number = Column(Integer, nullable=False, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    content_status = Column(String(64), nullable=False)
    signoff_status = Column(String(64), nullable=False)
    source_status = Column(String(64), nullable=False)
    provenance_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class SignedTextbookSourceFragment(Base):
    """Stable textbook fragments shared by one or more signed knowledge points."""

    __tablename__ = "signed_textbook_source_fragment"

    fragment_id = Column(String(128), primary_key=True)
    source_id = Column(String(128), nullable=False)
    lesson_number = Column(Integer, nullable=False, index=True)
    book_id = Column(String(32), nullable=False)
    lesson_id = Column(String(64), nullable=False)
    excerpt = Column(Text, nullable=False)
    textbook_pages_json = Column(Text, nullable=False)
    pdf_pages_json = Column(Text, nullable=False)
    source_type = Column(String(64), nullable=False)
    verification_status = Column(String(128), nullable=False)
    support_score = Column(Float, nullable=True)
    historical_source_anomaly = Column(Boolean, nullable=False, default=False)
    metadata_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class SignedTextbookKnowledgeSource(Base):
    """Normalized knowledge-fragment-parent-source many-to-many association."""

    __tablename__ = "signed_textbook_knowledge_source"

    knowledge_id = Column(
        String(64),
        ForeignKey("signed_textbook_knowledge.knowledge_id"),
        primary_key=True,
    )
    fragment_id = Column(
        String(128),
        ForeignKey("signed_textbook_source_fragment.fragment_id"),
        primary_key=True,
        index=True,
    )
    source_id = Column(String(128), primary_key=True, index=True)
    relation_status = Column(String(64), nullable=False, index=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class QuestionHistory(Base):
    """问答历史表模型，用于后续保存用户问题、回答和知识来源快照。"""

    __tablename__ = "question_history"

    id = Column(Integer, primary_key=True, index=True, comment="问答记录ID")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True, comment="用户ID")
    question = Column(Text, nullable=False, comment="用户问题")
    answer = Column(Text, nullable=False, comment="系统回答")
    sources = Column(Text, nullable=True, comment="知识来源快照，使用 JSON 字符串存储")
    created_at = Column(DateTime, server_default=func.now(), index=True, comment="创建时间")
