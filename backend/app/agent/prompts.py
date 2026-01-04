"""
Centralized prompt templates for a robust, analytical document Q&A agent.
Includes backward compatibility for existing agent code.
"""
from typing import List, Optional


# ============================================================
# PASS 1 — ANALYTICAL ANSWER PROMPT
# ============================================================

class AnswerPrompt:
    """
    Generates an analytical, grounded answer from document context.
    Allows synthesis and evaluation (critical for investment questions).
    """

    SYSTEM_MESSAGE = (
        "You are an expert analytical assistant answering questions STRICTLY "
        "based on the provided document context."
    )

    @staticmethod
    def build(question: str, chunks: List[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page_num = chunk.get("page_number", "?")
            text = chunk.get("text", "")
            context_parts.append(f"[Chunk {i}, Page {page_num}]:\n{text}\n")

        context = "\n".join(context_parts)

        return f"""
You are an expert analytical assistant.

IMPORTANT RULES:
- Use ONLY the provided document context.
- DO NOT use external knowledge.
- You MAY analyze, synthesize, and evaluate information in the document.
- The document may not state answers explicitly.
- For investment, safety, or risk questions:
  - Provide a balanced assessment.
  - Mention confidence scores, risks, assumptions, and uncertainties if present.
  - Do NOT claim certainty or guarantees.
- Do NOT say "the document does not contain information" if relevant signals exist.
- Only refuse if the document is completely unrelated.

STYLE:
- Analytical
- Cautious
- Grounded
- Exactly 3 short paragraphs
- No page numbers in the answer

Document Context:
{context}

Question:
{question}

Answer:
"""


# ============================================================
# PASS 2 — STRICT HALLUCINATION CHECK
# ============================================================

class HallucinationCheckPrompt:
    """
    Strict grounding validator to detect hallucinations.
    """

    SYSTEM_MESSAGE = (
        "You are a strict grounding validator. Detect unsupported claims."
    )

    @staticmethod
    def build(question: str, answer: str, chunks: List[dict]) -> str:
        context = "\n".join(chunk.get("text", "") for chunk in chunks)

        return f"""
You are a strict validator.

Task:
- Determine if the answer is fully supported by the document context.
- Analytical inference is allowed ONLY if based on document signals.
- NO external facts or assumptions allowed.

Question:
{question}

Document Context:
{context}

Answer:
{answer}

Respond ONLY in JSON:
{{
  "grounded": true or false,
  "reason": "short explanation"
}}
"""


# ============================================================
# PASS 3 — SAFE REWRITE / REFUSAL (ONLY IF NEEDED)
# ============================================================

class SafeRewritePrompt:
    """
    Conservative fallback if hallucination is detected.
    """

    SYSTEM_MESSAGE = (
        "You are a safety-focused assistant correcting unsupported answers."
    )

    @staticmethod
    def build(question: str, chunks: List[dict]) -> str:
        context = "\n".join(chunk.get("text", "") for chunk in chunks)

        return f"""
Provide a SAFE, CONSERVATIVE answer.

Rules:
- Use ONLY the document context.
- Avoid inference unless clearly supported.
- If a cautious analytical answer is possible, provide it.
- Otherwise, clearly refuse.

Document Context:
{context}

Question:
{question}

Safe Answer:
"""


# ============================================================
# ✅ BACKWARD-COMPATIBLE VALIDATION PROMPT (RESTORED)
# ============================================================

class ValidationPrompt:
    """
    Backward-compatible strict validation prompt.
    Used by existing agent/tools.py code.
    """

    SYSTEM_MESSAGE = "You are a strict validator. Respond with only YES or NO."

    @staticmethod
    def build(question: str, answer: str, context: str) -> str:
        return f"""
You are a validator.

Determine whether the ANSWER is supported by the DOCUMENT CONTEXT.

Rules:
- Analytical synthesis IS allowed if derived from document signals.
- Reject only if the answer introduces facts not supported by the document.
- If the answer is clearly grounded, respond YES.
- If the answer hallucinates or is unrelated, respond NO.

Question:
{question}

Document Context:
{context}

Answer:
{answer}

Respond with only "YES" or "NO":
"""


# ============================================================
# LENIENT QUESTION RELEVANCE PROMPT
# ============================================================

class RelevancePrompt:
    """
    Lenient question relevance validator.
    """

    SYSTEM_MESSAGE = (
        "You are a lenient validator. Reject only clearly unrelated questions."
    )

    @staticmethod
    def build(question: str, document_summary: Optional[str] = None) -> str:
        prompt = f"""
You are a relevance validator.

Question:
{question}
"""

        if document_summary:
            prompt += f"\nDocument Preview:\n{document_summary}\n"

        prompt += """
Rules:
- Be lenient.
- Allow analytical and evaluative questions.
- Reject ONLY clearly unrelated questions.

Respond in JSON:
{
  "is_relevant": true or false,
  "confidence": number,
  "reason": string
}
"""
        return prompt
