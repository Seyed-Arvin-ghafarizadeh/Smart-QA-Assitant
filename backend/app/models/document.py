"""Document data models."""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    page_number: int
    chunk_index: int
    document_id: str


@dataclass
class Document:
    """Represents a processed document."""

    document_id: str
    filename: str
    total_chunks: int
    total_pages: int
    chunks: Optional[List[Chunk]] = None

