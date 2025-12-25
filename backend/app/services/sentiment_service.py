"""Sentiment and relevance analysis service for questions."""
import time
from typing import Dict, Optional

from app.services.llm_service import LLMService
from app.utils.logger import logger


class SentimentService:
    """Service for analyzing question sentiment and relevance to documents."""

    def __init__(self, llm_service: LLMService):
        """
        Initialize sentiment service.

        Args:
            llm_service: LLM service instance for analysis
        """
        self.llm_service = llm_service

    async def analyze_question_relevance(
        self, question: str, document_summary: Optional[str] = None
    ) -> Dict:
        """
        Analyze if a question is relevant to the document content.
        
        This performs a LENIENT check to ensure questions are not completely unrelated.
        The goal is to only reject clearly off-topic questions while allowing
        most document-related questions to pass.

        Args:
            question: User's question
            document_summary: Optional summary of document content for context

        Returns:
            Dictionary with:
                - is_relevant: bool
                - sentiment: str (positive, neutral, negative, inappropriate)
                - confidence: float (0-1)
                - reason: str (explanation)
        """
        start_time = time.time()

        # Build analysis prompt - LENIENT approach
        analysis_prompt = f"""You are a helpful validator for document Q&A systems. Your job is to check if a question COULD POSSIBLY be related to the document content.

Question: {question}
"""

        if document_summary:
            analysis_prompt += f"\nDocument Content Preview:\n{document_summary}\n"

        analysis_prompt += """
IMPORTANT RULES - Be LENIENT:
- Respond with RELEVANT: YES if the question COULD be about the document content
- Respond with RELEVANT: YES if the question asks about topics, concepts, or information that might be in documents
- Respond with RELEVANT: YES even if you're not 100% sure - give the benefit of the doubt
- Only respond with RELEVANT: NO for questions that are CLEARLY unrelated like:
  * "What's the weather today?"
  * "Tell me a joke"
  * "What time is it?"
  * Questions about completely different topics with no possible connection to the document
- If the question mentions anything that could be in a document, respond YES
- When in doubt, respond YES

Respond in this EXACT format (one per line):
RELEVANT: YES or NO
SENTIMENT: POSITIVE, NEUTRAL, NEGATIVE, or INAPPROPRIATE
CONFIDENCE: 0.0 to 1.0 (as decimal)
REASON: Brief explanation (one sentence)
"""

        try:
            response = await self.llm_service.client.chat.completions.create(
                model=self.llm_service.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a lenient validator. When in doubt, allow the question. Only reject clearly unrelated questions.",
                    },
                    {"role": "user", "content": analysis_prompt},
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=150,
            )

            result_text = response.choices[0].message.content.strip()
            
            # Parse response - DEFAULT TO TRUE (relevant) for safety
            is_relevant = True  # Changed from False - be permissive by default
            sentiment = "neutral"
            confidence = 0.5
            reason = "Analysis completed"

            lines = result_text.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("RELEVANT:"):
                    # Only set to False if explicitly NO
                    is_relevant = "NO" not in line.upper()
                elif line.startswith("SENTIMENT:"):
                    sentiment = line.split(":", 1)[1].strip().lower()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        confidence = 0.5
                elif line.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            response_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Question relevance analysis completed",
                extra={
                    "question": question[:100],
                    "is_relevant": is_relevant,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "response_time_ms": response_time_ms,
                },
            )

            return {
                "is_relevant": is_relevant,
                "sentiment": sentiment,
                "confidence": confidence,
                "reason": reason,
                "response_time_ms": response_time_ms,
            }

        except Exception as e:
            logger.error(f"Error analyzing question relevance: {str(e)}", exc_info=True)
            # On error, allow the question to not block users
            return {
                "is_relevant": True,  # Allow on error to not block users
                "sentiment": "neutral",
                "confidence": 0.5,
                "reason": f"Analysis error: {str(e)}",
                "response_time_ms": (time.time() - start_time) * 1000,
            }

    async def validate_question_sentiment(self, question: str) -> Dict:
        """
        Validate question sentiment and appropriateness.

        This is a quick check for inappropriate content.

        Args:
            question: User's question

        Returns:
            Dictionary with sentiment analysis results
        """
        # Quick sentiment check for inappropriate content
        # Use word boundaries to avoid false positives (e.g., "skills" containing "kill")
        inappropriate_keywords = [
            r"\bhate\b", r"\bkill\b", r"\bviolence\b", r"\billegal\b", r"\bdrug\b", r"\bweapon\b",
            r"\battack\b", r"\bharm\b", r"\bthreat\b", r"\babuse\b"
        ]

        import re
        question_lower = question.lower()
        has_inappropriate = any(re.search(keyword, question_lower) for keyword in inappropriate_keywords)

        # Debug logging
        if has_inappropriate:
            matched_keywords = [kw for kw in inappropriate_keywords if kw in question_lower]
            print(f"DEBUG: Question '{question}' flagged for keywords: {matched_keywords}")

        result = {
            "is_appropriate": not has_inappropriate,
            "sentiment": "inappropriate" if has_inappropriate else "neutral",
            "confidence": 0.9 if has_inappropriate else 0.8,
            "reason": "Question contains potentially inappropriate content" if has_inappropriate else "Question appears appropriate",
        }

        print(f"DEBUG: Sentiment analysis result: {result}")
        return result

