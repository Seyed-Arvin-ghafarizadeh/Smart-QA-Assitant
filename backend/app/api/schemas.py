"""Pydantic schemas for API requests and responses."""
import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class UploadResponse(BaseModel):
    """Response schema for document upload."""

    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    filename: str = Field(..., description="Original filename")
    total_chunks: int = Field(..., description="Number of text chunks created")
    total_pages: int = Field(..., description="Number of pages in the document")
    message: str = Field(default="Document uploaded and processed successfully")


class AskRequest(BaseModel):
    """Request schema for asking questions."""

    question: str = Field(..., min_length=1, description="User's question")

    @field_validator("question")
    @classmethod
    def clean_question(cls, v: str) -> str:
        """
        Clean question by removing invalid control characters.
        
        Args:
            v: Raw question string
            
        Returns:
            Cleaned question string
        """
        if not isinstance(v, str):
            v = str(v)
        
        # Remove control characters except newline, tab, and carriage return
        # Keep: \n (0x0A), \t (0x09), \r (0x0D)
        # Remove: other control characters (0x00-0x08, 0x0B-0x0C, 0x0E-0x1F)
        cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', v)
        
        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()
        
        if not cleaned:
            raise ValueError("Question cannot be empty after cleaning")
        
        return cleaned


class RelevantChunk(BaseModel):
    """Schema for a relevant chunk with metadata."""

    text: str = Field(..., description="Chunk text content")
    page_number: int = Field(..., description="Page number where chunk appears")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")


class AskResponse(BaseModel):
    """Response schema for question answering."""

    answer: str = Field(..., description="LLM-generated answer")
    relevant_chunks: List[RelevantChunk] = Field(..., description="Retrieved relevant chunks")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    token_usage: Optional[dict] = Field(None, description="Token usage statistics")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")


class MetricsResponse(BaseModel):
    """Response schema for metrics endpoint."""

    metrics: dict = Field(..., description="Prometheus-compatible metrics")

