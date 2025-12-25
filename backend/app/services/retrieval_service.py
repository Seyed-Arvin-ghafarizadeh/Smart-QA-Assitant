"""Retrieval service orchestrating the RAG pipeline."""
import time
from typing import Dict, List, Optional

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStore
from app.services.sentiment_service import SentimentService
from app.utils.logger import logger


class RetrievalService:
    """Orchestrates the RAG pipeline: retrieval and answer generation."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
        sentiment_service: Optional[SentimentService] = None,
        similarity_max_threshold: float = 0.60,
        similarity_avg_threshold: float = 0.50,
        similarity_min_score: float = 0.50,
        similarity_high_count_min: int = 1,
    ):
        """
        Initialize retrieval service.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Service for vector storage and retrieval
            llm_service: Service for LLM answer generation
            sentiment_service: Optional sentiment analysis service
            similarity_max_threshold: Maximum similarity threshold (default: 0.60)
            similarity_avg_threshold: Average similarity threshold (default: 0.50)
            similarity_min_score: Minimum score for "high similarity" chunks (default: 0.50)
            similarity_high_count_min: Minimum number of chunks with high similarity (default: 1)
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.sentiment_service = sentiment_service or SentimentService(llm_service)
        self.similarity_max_threshold = similarity_max_threshold
        self.similarity_avg_threshold = similarity_avg_threshold
        self.similarity_min_score = similarity_min_score
        self.similarity_high_count_min = similarity_high_count_min

    async def retrieve_and_answer(
        self, question: str, top_k: int = 5
    ) -> Dict:
        """
        End-to-end RAG pipeline: retrieve relevant chunks across all documents and generate answer.
        
        This pipeline includes:
        1. Sentiment analysis and question relevance check
        2. Stricter similarity threshold validation
        3. Multi-level relevance validation
        4. Answer generation only if question passes all checks

        Args:
            question: User's question
            top_k: Number of chunks to retrieve (default: 5)

        Returns:
            Dictionary with answer, relevant_chunks, and metadata
        """
        start_time = time.time()

        # Step 0: Sentiment analysis and question relevance check
        sentiment_result = await self.sentiment_service.validate_question_sentiment(question)
        
        if not sentiment_result.get("is_appropriate", True):
            logger.warning(
                f"Question flagged as inappropriate: {question[:100]}",
                extra={"sentiment": sentiment_result.get("sentiment")}
            )
            total_time = time.time() - start_time
            return {
                "answer": "I'm sorry, but I cannot answer this question. Please ask a professional question related to the content in the uploaded documents.",
                "relevant_chunks": [],
                "confidence": 0.0,
                "token_usage": None,
                "response_time_ms": total_time * 1000,
                "cannot_answer": True,
                "sentiment": sentiment_result.get("sentiment", "inappropriate"),
                "is_relevant": False,
            }

        # Step 1: Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(question)

        # Step 2: Retrieve similar chunks across all documents
        retrieved_chunks = self.vector_store.retrieve_similar_across_all(
            query_embedding, top_k=top_k
        )

        if not retrieved_chunks:
            raise ValueError(
                "No matching documents found. This could mean: "
                "(1) No documents have been uploaded, or "
                "(2) The embedding model was changed and old documents need to be re-uploaded. "
                "Check server logs for more details."
            )

        # Log similarity scores
        similarity_scores = [chunk["similarity_score"] for chunk in retrieved_chunks]
        average_similarity = sum(similarity_scores) / len(similarity_scores)
        max_similarity = max(similarity_scores)
        min_similarity = min(similarity_scores)
        
        logger.info(
            f"Retrieved {len(retrieved_chunks)} chunks across all documents",
            extra={
                "similarity_scores": similarity_scores,
                "retrieval_score": average_similarity,
                "max_similarity": max_similarity,
                "min_similarity": min_similarity,
            },
        )

        # Step 2.5: Similarity check with configurable thresholds
        # Multi-level validation using configurable thresholds
        # 1. Maximum similarity must be above threshold
        # 2. Average similarity must be above threshold
        # 3. At least N chunks must have similarity > min_score
        
        high_similarity_count = sum(1 for score in similarity_scores if score > self.similarity_min_score)
        
        max_check = max_similarity >= self.similarity_max_threshold
        avg_check = average_similarity >= self.similarity_avg_threshold
        count_check = high_similarity_count >= self.similarity_high_count_min
        
        similarity_check_passed = max_check and avg_check and count_check
        
        # Enhanced logging with detailed check results
        logger.info(
            f"Similarity check results: "
            f"max={max_similarity:.3f} {'✓' if max_check else '✗'} (threshold: {self.similarity_max_threshold}), "
            f"avg={average_similarity:.3f} {'✓' if avg_check else '✗'} (threshold: {self.similarity_avg_threshold}), "
            f"high_sim_count={high_similarity_count} {'✓' if count_check else '✗'} (need >= {self.similarity_high_count_min} with score > {self.similarity_min_score}), "
            f"overall={'PASSED' if similarity_check_passed else 'FAILED'}",
            extra={
                "similarity_check_passed": similarity_check_passed,
                "max_similarity": max_similarity,
                "max_threshold": self.similarity_max_threshold,
                "avg_similarity": average_similarity,
                "avg_threshold": self.similarity_avg_threshold,
                "high_similarity_count": high_similarity_count,
                "min_score": self.similarity_min_score,
                "required_count": self.similarity_high_count_min,
            }
        )
        
        if not similarity_check_passed:
            failed_checks = []
            if not max_check:
                failed_checks.append(f"max similarity {max_similarity:.3f} < {self.similarity_max_threshold}")
            if not avg_check:
                failed_checks.append(f"avg similarity {average_similarity:.3f} < {self.similarity_avg_threshold}")
            if not count_check:
                failed_checks.append(f"only {high_similarity_count} chunks with score > {self.similarity_min_score} (need >= {self.similarity_high_count_min})")
            
            logger.info(
                f"Similarity check failed. Failed conditions: {', '.join(failed_checks)}. "
                f"Question may not be answerable from document."
            )
            # Return response indicating question cannot be answered
            total_time = time.time() - start_time
            return {
                "answer": "I'm sorry, but I cannot answer this question based on the uploaded document. Please ask a question that is more directly related to the content in the document.",
                "relevant_chunks": [
                    {
                        "text": chunk["text"],
                        "page_number": chunk["page_number"],
                        "chapter_number": chunk.get("chapter_number"),
                        "similarity_score": chunk["similarity_score"],
                        "document_id": chunk.get("document_id"),
                    }
                    for chunk in retrieved_chunks
                ],
                "confidence": average_similarity,
                "token_usage": None,
                "response_time_ms": total_time * 1000,
                "cannot_answer": True,
                "sentiment": sentiment_result.get("sentiment", "neutral"),
                "is_relevant": False,
            }

        # Step 2.6: Additional LLM-based relevance check
        # Get a summary of retrieved chunks for relevance analysis
        chunk_texts = [chunk["text"][:200] for chunk in retrieved_chunks[:3]]  # First 3 chunks
        document_summary = "\n".join(chunk_texts)
        
        relevance_analysis = await self.sentiment_service.analyze_question_relevance(
            question, document_summary
        )
        
        if not relevance_analysis.get("is_relevant", True):
            logger.info(
                f"LLM relevance check failed: {relevance_analysis.get('reason', 'Not relevant')}",
                extra={
                    "sentiment": relevance_analysis.get("sentiment"),
                    "confidence": relevance_analysis.get("confidence"),
                }
            )
            total_time = time.time() - start_time
            return {
                "answer": "I'm sorry, but this question does not appear to be related to the content in the uploaded documents. Please ask a question that is directly related to the document content.",
                "relevant_chunks": [
                    {
                        "text": chunk["text"],
                        "page_number": chunk["page_number"],
                        "chapter_number": chunk.get("chapter_number"),
                        "similarity_score": chunk["similarity_score"],
                        "document_id": chunk.get("document_id"),
                    }
                    for chunk in retrieved_chunks
                ],
                "confidence": average_similarity,
                "token_usage": None,
                "response_time_ms": total_time * 1000,
                "cannot_answer": True,
                "sentiment": relevance_analysis.get("sentiment", "neutral"),
                "is_relevant": False,
                "relevance_reason": relevance_analysis.get("reason", ""),
            }

        # Step 3: Generate answer using LLM (only if all checks passed)
        llm_result = await self.llm_service.generate_answer(question, retrieved_chunks)
        answer = llm_result["answer"]
        
        # Check if LLM already returned a "cannot answer" message
        cannot_answer_phrases = [
            "cannot answer",
            "cannot be found",
            "not in the context",
            "not available in",
            "not provided",
            "please ask a question related",
        ]
        answer_lower = answer.lower()
        llm_says_cannot_answer = any(phrase in answer_lower for phrase in cannot_answer_phrases)
        
        # Step 4: STRICT validation that the answer is actually based on the retrieved chunks
        # Skip validation if LLM already says it cannot answer
        if not llm_says_cannot_answer:
            is_answerable = await self.llm_service.validate_answer_relevance(
                question, answer, retrieved_chunks
            )
            
            if not is_answerable:
                logger.info("STRICT LLM validation determined answer is not based on document content")
                answer = "I'm sorry, but I cannot answer this question based on the uploaded document. Please ask a question that is directly related to the content in the document."
                llm_result["answer"] = answer
                llm_says_cannot_answer = True

        total_time = time.time() - start_time

        # Format response with sentiment and relevance information
        response = {
            "answer": answer,
            "relevant_chunks": [
                {
                    "text": chunk["text"],
                    "page_number": chunk["page_number"],
                    "chapter_number": chunk.get("chapter_number"),
                    "similarity_score": chunk["similarity_score"],
                    "document_id": chunk.get("document_id"),  # Include document ID for frontend
                }
                for chunk in retrieved_chunks
            ],
            "confidence": average_similarity,
            "token_usage": llm_result.get("token_usage"),
            "response_time_ms": total_time * 1000,
            "sentiment": relevance_analysis.get("sentiment", sentiment_result.get("sentiment", "neutral")),
            "is_relevant": True,  # Passed all checks
            "similarity_metrics": {
                "max_similarity": max_similarity,
                "avg_similarity": average_similarity,
                "min_similarity": min_similarity,
                "high_similarity_count": high_similarity_count,
            },
        }

        # Add cannot_answer flag if LLM says it cannot answer
        if llm_says_cannot_answer:
            response["cannot_answer"] = True
            response["is_relevant"] = False

        logger.info(
            f"RAG pipeline completed",
            extra={
                "llm_response_time": llm_result.get("response_time_ms"),
                "total_time_ms": total_time * 1000,
                "token_usage": llm_result.get("token_usage"),
                "answer_length": len(answer),
                "sentiment": response.get("sentiment"),
                "is_relevant": response.get("is_relevant"),
                "similarity_metrics": response.get("similarity_metrics"),
            },
        )

        return response

