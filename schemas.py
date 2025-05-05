import uuid
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from fastapi_users import schemas


class UserRead(schemas.BaseUser[uuid.UUID]):
    pass

class UserCreate(schemas.BaseUserCreate):
    pass

class UserUpdate(schemas.BaseUserUpdate):
    pass


class TranscriptionSegment(BaseModel):
    start_time: str
    end_time: str
    speaker_id: int
    transcript: str


class TranscriptionSegments(BaseModel):
     transcriptions: List[TranscriptionSegment]


class TranscriptionApiResponse(BaseModel):
     transcriptions: List[TranscriptionSegment]
     summarized_text: Optional[str] = None


class TranscriptionHistoryItem(BaseModel):
    id: uuid.UUID
    filename: str
    created_at: datetime
    status: str

    class Config:
        from_attributes = True


class TranscriptionDetail(TranscriptionHistoryItem):
    result: Optional[TranscriptionSegments] = None
    summarized_text: Optional[str] = None

    class Config:
        from_attributes = True