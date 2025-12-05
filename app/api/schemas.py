from typing import List, Optional
from pydantic import BaseModel


class PhonemeScoreDetail(BaseModel):
    phoneme: str
    quality_score: int
    duration_score: int


class PredictionResponse(BaseModel):
    success: bool
    word: Optional[str] = None
    original_transcript: str
    details: List[PhonemeScoreDetail]


class ErrorResponse(BaseModel):
    detail: str
