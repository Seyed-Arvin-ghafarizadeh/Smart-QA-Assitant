"""
Retrieval service orchestrating a robust RAG pipeline.
"""
import time
from typing import Dict, List, Optional

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStore
from app.services.sentiment_service import SentimentService
from app.utils.logger import logger


class RetrievalService:
    """
    Orchestrates retrieval + validation + answer generation for RAG.
    """

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
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.sentiment_service = sentiment_service or SentimentService(llm_service)

        # Similarity thresholds
        self.similarity_max_threshold = similarity_max_threshold
        self.similarity_avg_threshold = similarity_avg_threshold
        self.similarity_min_score = similarity_min_score
        self.similarity_high_count_min = similarity_high_count_min

    # ------------------------------------------------------------------
    # Similarity Evaluation
    # ------------------------------------------------------------------

    def _evaluate_similarity(self, scores: List[float]) -> Dict:
        max_sim = max(scores)
        avg_sim = sum(scores) / len(scores)
        min_sim = min(scores)
        high_count = sum(1 for s in scores if s >= self.similarity_min_score)

        checks = {
            "max_check": max_sim >= self.similarity_max_threshold,
            "avg_check": avg_sim >= self.similarity_avg_threshold,
            "count_check": high_count >= self.similarity_high_count_min,
        }

        return {
            "passed": all(checks.values()),
            "metrics": {
                "max_similarity": max_sim,
                "avg_similarity": avg_sim,
                "min_similarity": min_sim,
                "high_similarity_count": high_count,
            },
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------

    async def retrieve_and_answer(self, question: str, top_k: int = 5) -> Dict:
        start_time = time.time()

        # --------------------------------------------------------------
        # 0. Sentiment / Safety Check (HARD STOP)
        # --------------------------------------------------------------
        sentiment = await self.sentiment_service.validate_question_sentiment(question)

        if not sentiment.get("is_appropriate", True):
            return self._blocked_response(
                start_time,
                sentiment.get("sentiment", "inappropriate"),
            )

        # --------------------------------------------------------------
        # 1. Embedding + Retrieval
        # --------------------------------------------------------------
        query_embedding = self.embedding_service.generate_embedding(question)

        retrieved_chunks = self.vector_store.retrieve_similar_across_all(
            query_embedding, top_k=top_k
        )

        if not retrieved_chunks:
            logger.warning("No chunks retrieved")
            return self._cannot_answer(
                start_time,
                "No relevant document content found.",
                [],
            )

        similarity_scores = [c["similarity_score"] for c in retrieved_chunks]
        similarity_eval = self._evaluate_similarity(similarity_scores)

        logger.info(
            "retrieval_similarity_evaluation",
            extra={
                **similarity_eval["metrics"],
                **similarity_eval["checks"],
            },
        )

        # --------------------------------------------------------------
        # 2. Similarity Gate (HARD STOP)
        # --------------------------------------------------------------
        if not similarity_eval["passed"]:
            return self._cannot_answer(
                start_time,
                "Question is not sufficiently related to the document content.",
                retrieved_chunks,
                similarity_eval["metrics"],
            )

        # --------------------------------------------------------------
        # 3. LLM Relevance Check (SOFT GATE)
        # --------------------------------------------------------------
        preview_text = "\n".join(
            chunk["text"][:200] for chunk in retrieved_chunks[:3]
        )

        relevance = await self.sentiment_service.analyze_question_relevance(
            question,
            document_summary=preview_text,
        )

        if not relevance.get("is_relevant", True):
            return self._cannot_answer(
                start_time,
                relevance.get("reason", "Question not relevant"),
                retrieved_chunks,
                similarity_eval["metrics"],
                sentiment=relevance.get("sentiment", "neutral"),
            )

        # --------------------------------------------------------------
        # 4. Answer Generation
        # --------------------------------------------------------------
        llm_result = await self.llm_service.generate_answer(
            question, retrieved_chunks
        )

        answer = llm_result["answer"]
        answer_lower = answer.lower()

        cannot_answer_phrases = (
            "cannot answer",
            "not found in the document",
            "not provided",
            "not available",
        )

        llm_says_no = any(p in answer_lower for p in cannot_answer_phrases)

        # --------------------------------------------------------------
        # 5. STRICT Answer Validation (ONLY IF NEEDED)
        # --------------------------------------------------------------
        if not llm_says_no:
            valid = await self.llm_service.validate_answer_relevance(
                question, answer, retrieved_chunks
            )
            if not valid:
                llm_says_no = True
                answer = (
                    "I'm sorry, but I cannot answer this question based on "
                    "the uploaded documents."
                )

        # --------------------------------------------------------------
        # 6. Final Response
        # --------------------------------------------------------------
        total_time = (time.time() - start_time) * 1000

        response = {
            "answer": answer,
            "relevant_chunks": self._format_chunks(retrieved_chunks),
            "confidence": similarity_eval["metrics"]["avg_similarity"],
            "token_usage": llm_result.get("token_usage"),
            "response_time_ms": total_time,
            "sentiment": relevance.get("sentiment", "neutral"),
            "is_relevant": not llm_says_no,
            "similarity_metrics": similarity_eval["metrics"],
        }

        if llm_says_no:
            response["cannot_answer"] = True

        logger.info(
            "rag_pipeline_completed",
            extra={
                "response_time_ms": total_time,
                "answer_length": len(answer),
                "is_relevant": response["is_relevant"],
            },
        )

        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_chunks(self, chunks: List[Dict]) -> List[Dict]:
        return [
            {
                "text": c["text"],
                "page_number": c.get("page_number"),
                "chapter_number": c.get("chapter_number"),
                "similarity_score": c["similarity_score"],
                "document_id": c.get("document_id"),
            }
            for c in chunks
        ]

    def _cannot_answer(
        self,
        start_time: float,
        reason: str,
        chunks: List[Dict],
        similarity_metrics: Optional[Dict] = None,
        sentiment: str = "neutral",
    ) -> Dict:
        return {
            "answer": (
                "I'm sorry, but I cannot answer this question based on "
                "the uploaded documents."
            ),
            "relevant_chunks": self._format_chunks(chunks),
            "confidence": similarity_metrics.get("avg_similarity", 0.0)
            if similarity_metrics
            else 0.0,
            "token_usage": None,
            "response_time_ms": (time.time() - start_time) * 1000,
            "cannot_answer": True,
            "is_relevant": False,
            "sentiment": sentiment,
            "similarity_metrics": similarity_metrics,
            "reason": reason,
        }

    def _blocked_response(self, start_time: float, sentiment: str) -> Dict:
        return {
            "answer": (
                "I'm sorry, but I cannot help with this question. "
                "Please ask a professional question related to the documents."
            ),
            "relevant_chunks": [],
            "confidence": 0.0,
            "token_usage": None,
            "response_time_ms": (time.time() - start_time) * 1000,
            "cannot_answer": True,
            "is_relevant": False,
            "sentiment": sentiment,
        }
