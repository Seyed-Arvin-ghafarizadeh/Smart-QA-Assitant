"""
Sentiment and semantic relevance analysis service for questions.
"""
import time
import json
import re
from typing import Dict, Optional

import numpy as np

from app.services.llm_service import LLMService
from app.utils.logger import logger


class SentimentService:
    """
    Service for analyzing question sentiment and semantic relevance to documents.
    """

    SEMANTIC_THRESHOLD_RELEVANT = 0.20   # clearly relevant
    SEMANTIC_THRESHOLD_REJECT = 0.05     # clearly irrelevant

    INAPPROPRIATE_PATTERNS = [
        r"\b(kill|murder|assault|rape)\b",
        r"\b(hate|racist|sexist)\b",
        r"\b(bomb|weapon|gun|knife)\b",
        r"\b(drugs?|cocaine|heroin)\b",
        r"\b(attack|threat|harm|abuse)\b",
    ]

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ---------------------------------------------------------------------
    # Sentiment & Safety
    # ---------------------------------------------------------------------

    async def validate_question_sentiment(self, question: str) -> Dict:
        """
        Fast local check for inappropriate content.
        """
        question_lower = question.lower()

        is_inappropriate = any(
            re.search(pattern, question_lower)
            for pattern in self.INAPPROPRIATE_PATTERNS
        )

        return {
            "is_appropriate": not is_inappropriate,
            "sentiment": "inappropriate" if is_inappropriate else "neutral",
            "confidence": 0.9 if is_inappropriate else 0.8,
            "reason": (
                "Question contains potentially inappropriate content"
                if is_inappropriate
                else "Question appears appropriate"
            ),
        }

    # ---------------------------------------------------------------------
    # Semantic Relevance (Primary Signal)
    # ---------------------------------------------------------------------

    async def _semantic_relevance(
        self,
        question: str,
        document_embedding: list[float],
    ) -> Dict:
        question_embedding = await self.llm_service.get_embedding(question)
        similarity = self._cosine_similarity(question_embedding, document_embedding)

        return {
            "similarity_score": round(similarity, 3),
            "is_relevant": similarity >= self.SEMANTIC_THRESHOLD_RELEVANT,
            "confidence": min(1.0, similarity + 0.2),
        }

    # ---------------------------------------------------------------------
    # LLM Fallback (Borderline Cases Only)
    # ---------------------------------------------------------------------

    async def _llm_relevance_fallback(
        self,
        question: str,
        document_summary: Optional[str],
    ) -> Dict:
        prompt = f"""
You are a LENIENT validator for document Q&A systems.

Return ONLY valid JSON using this schema:
{{
  "is_relevant": boolean,
  "sentiment": "positive" | "neutral" | "negative" | "inappropriate",
  "confidence": number,
  "reason": string
}}

Rules:
- Be lenient
- When in doubt, return is_relevant = true
- Reject ONLY clearly unrelated questions

Question:
{question}
"""

        if document_summary:
            prompt += f"\nDocument Preview:\n{document_summary}\n"

        response = await self.llm_service.client.chat.completions.create(
            model=self.llm_service.model,
            messages=[
                {"role": "system", "content": "You are a lenient document relevance validator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )

        content = response.choices[0].message.content.strip()
        return json.loads(content)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    async def analyze_question_relevance(
        self,
        question: str,
        document_summary: Optional[str] = None,
        document_embedding: Optional[list[float]] = None,
    ) -> Dict:
        """
        Analyze question sentiment and relevance using semantic search + LLM fallback.
        """
        start_time = time.time()

        try:
            # 1. Sentiment / Safety
            sentiment_result = await self.validate_question_sentiment(question)
            if not sentiment_result["is_appropriate"]:
                return {
                    "is_relevant": False,
                    **sentiment_result,
                    "response_time_ms": (time.time() - start_time) * 1000,
                }

            # 2. Semantic relevance (preferred)
            if document_embedding:
                semantic = await self._semantic_relevance(
                    question, document_embedding
                )

                if semantic["similarity_score"] >= self.SEMANTIC_THRESHOLD_RELEVANT:
                    return {
                        "is_relevant": True,
                        "sentiment": "neutral",
                        "confidence": semantic["confidence"],
                        "reason": "Semantically relevant to document",
                        "similarity_score": semantic["similarity_score"],
                        "response_time_ms": (time.time() - start_time) * 1000,
                    }

                if semantic["similarity_score"] < self.SEMANTIC_THRESHOLD_REJECT:
                    return {
                        "is_relevant": False,
                        "sentiment": "neutral",
                        "confidence": 0.9,
                        "reason": "Clearly unrelated by semantic similarity",
                        "similarity_score": semantic["similarity_score"],
                        "response_time_ms": (time.time() - start_time) * 1000,
                    }

            # 3. LLM fallback (borderline only)
            llm_result = await self._llm_relevance_fallback(
                question, document_summary
            )

            llm_result["response_time_ms"] = (
                time.time() - start_time
            ) * 1000

            return llm_result

        except Exception as e:
            logger.error("Question analysis failed", exc_info=True)
            return {
                "is_relevant": True,  # fail-open
                "sentiment": "neutral",
                "confidence": 0.5,
                "reason": f"Fallback due to error: {str(e)}",
                "response_time_ms": (time.time() - start_time) * 1000,
            }
