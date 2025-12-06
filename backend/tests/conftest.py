"""Pytest configuration and fixtures."""
import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = Mock(spec=EmbeddingService)
    service.generate_embedding = Mock(return_value=[0.1] * 384)
    service.generate_embeddings = Mock(return_value=[[0.1] * 384] * 3)
    return service


@pytest.fixture
def mock_vector_store(temp_dir):
    """Mock vector store."""
    return VectorStore(db_path=temp_dir)


@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    service = Mock(spec=LLMService)
    service.generate_answer = AsyncMock(
        return_value={
            "answer": "This is a test answer.",
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "response_time_ms": 500.0,
        }
    )
    return service


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    from app.models.document import Chunk

    return [
        Chunk(
            text="This is the first chunk of text.",
            page_number=1,
            chunk_index=0,
            document_id="test-doc-1",
        ),
        Chunk(
            text="This is the second chunk of text.",
            page_number=1,
            chunk_index=1,
            document_id="test-doc-1",
        ),
        Chunk(
            text="This is the third chunk of text.",
            page_number=2,
            chunk_index=2,
            document_id="test-doc-1",
        ),
    ]


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing (as bytes)."""
    # This is a minimal PDF structure for testing
    # In real tests, you'd use an actual PDF file
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"

