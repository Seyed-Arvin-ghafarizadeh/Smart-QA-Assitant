"""Validation logic for questions, answers, and outputs."""
import re
from typing import Dict, List

from app.utils.logger import logger


class QuestionValidator:
    """Validates questions for appropriateness and sentiment."""

    INAPPROPRIATE_KEYWORDS = [
        r"\bhate\b",
        r"\bkill\b",
        r"\bviolence\b",
        r"\billegal\b",
        r"\bdrug\b",
        r"\bweapon\b",
        r"\battack\b",
        r"\bharm\b",
        r"\bthreat\b",
        r"\babuse\b",
    ]

    def validate_sentiment(self, question: str) -> Dict:
        """
        Validate question sentiment and appropriateness.

        Args:
            question: User's question

        Returns:
            Dictionary with is_appropriate, sentiment, confidence, reason
        """
        question_lower = question.lower()
        has_inappropriate = any(
            re.search(keyword, question_lower) for keyword in self.INAPPROPRIATE_KEYWORDS
        )

        result = {
            "is_appropriate": not has_inappropriate,
            "sentiment": "inappropriate" if has_inappropriate else "neutral",
            "confidence": 0.9 if has_inappropriate else 0.8,
            "reason": (
                "Question contains potentially inappropriate content"
                if has_inappropriate
                else "Question appears appropriate"
            ),
        }

        if has_inappropriate:
            logger.warning(
                f"Question flagged as inappropriate: {question[:100]}",
                extra={"sentiment": result["sentiment"]},
            )

        return result


class AnswerValidator:
    """Validates answer quality and relevance."""

    CANNOT_ANSWER_PHRASES = [
        "cannot answer",
        "cannot be found",
        "not in the context",
        "not available in",
        "not provided",
        "please ask a question related",
    ]

    def check_cannot_answer(self, answer: str) -> bool:
        """
        Check if answer indicates it cannot answer.

        Args:
            answer: Generated answer

        Returns:
            True if answer says it cannot answer
        """
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in self.CANNOT_ANSWER_PHRASES)

    def validate_similarity_thresholds(
        self,
        similarity_scores: List[float],
        max_threshold: float,
        avg_threshold: float,
        min_score: float,
        high_count_min: int,
    ) -> Dict:
        """
        Validate similarity scores against thresholds.

        Args:
            similarity_scores: List of similarity scores
            max_threshold: Maximum similarity threshold
            avg_threshold: Average similarity threshold
            min_score: Minimum score for "high similarity"
            high_count_min: Minimum number of chunks with high similarity

        Returns:
            Dictionary with passed, failed_checks, metrics
        """
        if not similarity_scores:
            return {
                "passed": False,
                "failed_checks": ["No similarity scores"],
                "metrics": {},
            }

        max_similarity = max(similarity_scores)
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        min_similarity = min(similarity_scores)
        high_similarity_count = sum(1 for score in similarity_scores if score > min_score)

        max_check = max_similarity >= max_threshold
        avg_check = avg_similarity >= avg_threshold
        count_check = high_similarity_count >= high_count_min

        passed = max_check and avg_check and count_check

        failed_checks = []
        if not max_check:
            failed_checks.append(
                f"max similarity {max_similarity:.3f} < {max_threshold}"
            )
        if not avg_check:
            failed_checks.append(
                f"avg similarity {avg_similarity:.3f} < {avg_threshold}"
            )
        if not count_check:
            failed_checks.append(
                f"only {high_similarity_count} chunks with score > {min_score} (need >= {high_count_min})"
            )

        metrics = {
            "max_similarity": max_similarity,
            "avg_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "high_similarity_count": high_similarity_count,
        }

        logger.info(
            f"Similarity check: {'PASSED' if passed else 'FAILED'}",
            extra={
                "similarity_check_passed": passed,
                "metrics": metrics,
                "failed_checks": failed_checks,
            },
        )

        return {
            "passed": passed,
            "failed_checks": failed_checks,
            "metrics": metrics,
        }


class OutputValidator:
    """Validates output format and structure."""

    def validate_response(self, response: Dict) -> Dict:
        """
        Validate response structure.

        Args:
            response: Response dictionary

        Returns:
            Dictionary with is_valid, errors
        """
        errors = []
        required_fields = ["answer", "relevant_chunks"]

        for field in required_fields:
            if field not in response:
                errors.append(f"Missing required field: {field}")

        if "answer" in response and not isinstance(response["answer"], str):
            errors.append("Answer must be a string")

        if "relevant_chunks" in response and not isinstance(
            response["relevant_chunks"], list
        ):
            errors.append("Relevant chunks must be a list")

        is_valid = len(errors) == 0

        return {"is_valid": is_valid, "errors": errors}

