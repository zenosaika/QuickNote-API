import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID, GUID

from database import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    pass


class Transcription(Base):
    __tablename__ = "transcriptions"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), default="completed")

    result_json = Column(Text, nullable=True)
    summarized_text = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Transcription(id={self.id}, filename='{self.filename}', user_id='{self.user_id}')>"