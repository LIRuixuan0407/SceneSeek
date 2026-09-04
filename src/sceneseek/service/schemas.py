from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScanRequest(BaseModel):
    path: str = Field(min_length=1)


class BuildRequest(BaseModel):
    rebuild: bool = False


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=24, ge=1, le=100)
    media_type: Literal["image", "video"] | None = None


class FeedbackRequest(BaseModel):
    query_id: str = Field(min_length=1, max_length=128)
    media_id: str = Field(min_length=1, max_length=128)
    relevance: int = Field(ge=0, le=3)
    note: str | None = Field(default=None, max_length=2000)
