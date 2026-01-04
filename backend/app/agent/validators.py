"""
Robust validation logic for questions, answers, and outputs.
Backward-compatible with existing agent code.
"""
import re
from typing import Dict, List

from app.utils.logger import logger


# ============================================================
# Question Validator
# ============================================================

class QuestionValidator:
    INAPPROPRIATE_KEYWORDS = [
        r"\b(kill|murder|assault|rape)\b",
        r"\b(hate|racist|sexist)\b",
        r"\b(weapon|bomb|gun|knife)\b",
        r"\b(drug|cocaine|heroin)\b",
        r"\b(attack|threat|harm|abuse)\b",
    ]

    ANALYTICAL_INTENT_KEYWORDS = [
        "safe", "risk", "invest", "investment", "viable",
        "worth", "opportunity", "confidence", "potential",
    ]

    def validate_sentiment(self, question: str) -> Dict:
        question_lower = question.lower()

        has_inappropriate = any(
            re.search(p, question_lower) for p in self.INAPPROPRIATE_KEYWORDS
        )

        intent = (
            "analytical"
            if any(k in question_lower for k in self.ANALYTICAL_INTENT_KEYWORDS)
            else "factual"
        )

        return {
            "is_appropriate": not has_inappropriate,
            "sentiment": "inappropriate" if has_inappropriate else "neutral",
            "confidence": 0.9 if has_inappropriate else 0.85,
            "intent": intent,
            "reason": (
                "Inappropriate content detected"
                if has_inappropriate
                else f"Appropriate question ({intent} intent)"
            ),
        }


# ============================================================
# Answer Validator
# ============================================================

class AnswerValidator:
    HARD_CANNOT_ANSWER_PHRASES = [
        "cannot answer",
        "cannot be found",
        "not in the context",
        "not available",
        "not provided",
    ]

    SOFT_ANALYTICAL_SIGNALS = [
        "based on the document",
        "analysis suggests",
        "high confidence",
        "regulatory risk",
        "market volatility",
        "investment defensible",
    ]

    def check_cannot_answer(self, answer: str) -> bool:
        answer_lower = answer.lower()
        return any(p in answer_lower for p in self.HARD_CANNOT_ANSWER_PHRASES)

    def validate_answer_grounding(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict],
        question_intent: str,
    ) -> Dict:
        if self.check_cannot_answer(answer):
            return {"is_valid": False, "reason": "LLM explicitly refused"}

        answer_lower = answer.lower()
        doc_text = " ".join(c["text"].lower() for c in retrieved_chunks)

        # ANALYTICAL QUESTIONS → allow inference
        if question_intent == "analytical":
            signal_match = any(s in answer_lower for s in self.SOFT_ANALYTICAL_SIGNALS)
            doc_signal = any(
                k in doc_text for k in ["confidence", "risk", "regulatory", "market"]
            )

            return {
                "is_valid": signal_match or doc_signal,
                "reason": "Analytical answer grounded in document signals",
            }

        # FACTUAL QUESTIONS → strict
        strict_match = any(chunk["text"].lower() in answer_lower for chunk in retrieved_chunks)

        return {
            "is_valid": strict_match,
            "reason": "Factual grounding check",
        }

    # ------------------------------------------------------------------
    # ✅ BACKWARD-COMPATIBILITY FIX (THIS SOLVES YOUR ERROR)
    # ------------------------------------------------------------------
    def validate_similarity_thresholds(
        self,
        similarity_scores: List[float],
        max_threshold: float,
        avg_threshold: float,
        min_score: float,
        high_count_min: int,
    ) -> Dict:
        if not similarity_scores:
            return {"passed": False, "failed_checks": ["No scores"], "metrics": {}}

        max_sim = max(similarity_scores)
        avg_sim = sum(similarity_scores) / len(similarity_scores)
        min_sim = min(similarity_scores)
        high_count = sum(1 for s in similarity_scores if s >= min_score)

        passed = (
            max_sim >= max_threshold
            and avg_sim >= avg_threshold
            and high_count >= high_count_min
        )

        return {
            "passed": passed,
            "failed_checks": [] if passed else ["Similarity thresholds not met"],
            "metrics": {
                "max_similarity": max_sim,
                "avg_similarity": avg_sim,
                "min_similarity": min_sim,
                "high_similarity_count": high_count,
            },
        }


# ============================================================
# Output Validator
# ============================================================

class OutputValidator:
    REQUIRED_FIELDS = ["answer", "relevant_chunks"]

    def validate_response(self, response: Dict) -> Dict:
        errors = []

        for f in self.REQUIRED_FIELDS:
            if f not in response:
                errors.append(f"Missing field: {f}")

        if "answer" in response and not isinstance(response["answer"], str):
            errors.append("Answer must be string")

        if "relevant_chunks" in response and not isinstance(
            response["relevant_chunks"], list
        ):
            errors.append("Relevant chunks must be list")

        return {"is_valid": len(errors) == 0, "errors": errors}
