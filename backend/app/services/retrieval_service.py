"""Retrieval service orchestrating the RAG pipeline."""
import time
from typing import Dict, List

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStore
from app.utils.logger import logger


class RetrievalService:
    """Orchestrates the RAG pipeline: retrieval and answer generation."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
    ):
        """
        Initialize retrieval service.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Service for vector storage and retrieval
            llm_service: Service for LLM answer generation
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service

    async def retrieve_and_answer(
        self, question: str, top_k: int = 5
    ) -> Dict:
        """
        End-to-end RAG pipeline: retrieve relevant chunks across all documents and generate answer.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve (default: 5)

        Returns:
            Dictionary with answer, relevant_chunks, and metadata
        """
        start_time = time.time()

        # Step 1: Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(question)

        # Step 2: Retrieve similar chunks across all documents
        retrieved_chunks = self.vector_store.retrieve_similar_across_all(
            query_embedding, top_k=top_k
        )

        if not retrieved_chunks:
            raise ValueError("No documents found in the database. Please upload a document first.")

        # Log similarity scores
        similarity_scores = [chunk["similarity_score"] for chunk in retrieved_chunks]
        logger.info(
            f"Retrieved {len(retrieved_chunks)} chunks across all documents",
            extra={
                "similarity_scores": similarity_scores,
                "retrieval_score": sum(similarity_scores) / len(similarity_scores),
            },
        )

        # Step 3: Generate answer using LLM
        llm_result = await self.llm_service.generate_answer(question, retrieved_chunks)

        total_time = time.time() - start_time

        # Format response
        response = {
            "answer": llm_result["answer"],
            "relevant_chunks": [
                {
                    "text": chunk["text"],
                    "page_number": chunk["page_number"],
                    "similarity_score": chunk["similarity_score"],
                }
                for chunk in retrieved_chunks
            ],
            "confidence": sum(similarity_scores) / len(similarity_scores),
            "token_usage": llm_result.get("token_usage"),
            "response_time_ms": total_time * 1000,
        }

        logger.info(
            f"RAG pipeline completed",
            extra={
                "llm_response_time": llm_result.get("response_time_ms"),
                "total_time_ms": total_time * 1000,
                "token_usage": llm_result.get("token_usage"),
                "answer_length": len(llm_result["answer"]),
            },
        )

        return response

