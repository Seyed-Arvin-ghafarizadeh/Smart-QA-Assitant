"""Tests for service modules."""
import pytest
from unittest.mock import Mock, patch, AsyncMock

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService
from app.models.document import Chunk


class TestDocumentProcessor:
    """Tests for DocumentProcessor."""

    def test_clean_text(self):
        """Test text cleaning functionality."""
        from app.utils.text_cleaner import clean_text

        dirty_text = "This   has    multiple    spaces\n\n\n\nand newlines"
        cleaned = clean_text(dirty_text)
        assert "  " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_chunk_text(self, sample_chunks):
        """Test text chunking."""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        text = " ".join([chunk.text for chunk in sample_chunks])
        page_breaks = [len(sample_chunks[0].text), len(sample_chunks[0].text) + len(sample_chunks[1].text)]

        chunks = processor.chunk_text(text, page_breaks, "test-doc", "test.pdf")
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    def test_singleton_pattern(self):
        """Test that EmbeddingService uses singleton pattern."""
        service1 = EmbeddingService()
        service2 = EmbeddingService()
        assert service1 is service2

    def test_generate_embedding(self):
        """Test single embedding generation."""
        service = EmbeddingService()
        embedding = service.generate_embedding("Test text")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_generate_embeddings(self):
        """Test batch embedding generation."""
        service = EmbeddingService()
        texts = ["First text", "Second text", "Third text"]
        embeddings = service.generate_embeddings(texts)
        assert len(embeddings) == len(texts)
        assert all(len(emb) > 0 for emb in embeddings)


class TestVectorStore:
    """Tests for VectorStore."""

    def test_store_and_retrieve(self, temp_dir, sample_chunks):
        """Test storing and retrieving chunks."""
        store = VectorStore(db_path=temp_dir)
        embeddings = [[0.1] * 384] * len(sample_chunks)

        store.store_document(sample_chunks, embeddings)
        assert store.document_exists("test-doc-1")

        query_embedding = [0.1] * 384
        results = store.retrieve_similar(query_embedding, "test-doc-1", top_k=2)
        assert len(results) > 0
        assert "text" in results[0]
        assert "similarity_score" in results[0]

    def test_document_exists(self, temp_dir):
        """Test document existence check."""
        store = VectorStore(db_path=temp_dir)
        assert not store.document_exists("non-existent-doc")


class TestLLMService:
    """Tests for LLMService."""

    @pytest.mark.asyncio
    async def test_generate_answer(self):
        """Test LLM answer generation."""
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            service = LLMService(api_key="test-key")
            service.client = AsyncMock()

            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Test answer"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
            mock_response.raise_for_status = Mock()

            service.client.post = AsyncMock(return_value=mock_response)

            chunks = [{"text": "Test chunk", "page_number": 1}]
            result = await service.generate_answer("Test question", chunks)

            assert "answer" in result
            assert result["answer"] == "Test answer"
            assert "token_usage" in result

    def test_build_prompt(self):
        """Test prompt building."""
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            service = LLMService(api_key="test-key")
            chunks = [
                {"text": "First chunk", "page_number": 1},
                {"text": "Second chunk", "page_number": 2},
            ]
            prompt = service._build_prompt("Test question", chunks)

            assert "Test question" in prompt
            assert "First chunk" in prompt
            assert "Second chunk" in prompt
            assert "Page 1" in prompt


class TestRetrievalService:
    """Tests for RetrievalService."""

    @pytest.mark.asyncio
    async def test_retrieve_and_answer(self, mock_embedding_service, mock_vector_store, mock_llm_service, sample_chunks):
        """Test end-to-end retrieval and answer generation."""
        # Store chunks in vector store
        embeddings = [[0.1] * 384] * len(sample_chunks)
        mock_vector_store.store_document(sample_chunks, embeddings)

        service = RetrievalService(mock_embedding_service, mock_vector_store, mock_llm_service)

        result = await service.retrieve_and_answer("Test question", "test-doc-1", top_k=2)

        assert "answer" in result
        assert "relevant_chunks" in result
        assert len(result["relevant_chunks"]) > 0

