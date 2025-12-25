"""LLM service for DeepSeek API integration."""
import os
import time
from typing import Dict, List, Optional

from openai import AsyncOpenAI
import httpx

from app.utils.logger import logger


class LLMService:
    """Service for interacting with DeepSeek API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.deepseek.com/v1/chat/completions",
        model: str = "deepseek-chat",
    ):
        """
        Initialize LLM service.

        Args:
            api_key: DeepSeek API key (from env if not provided)
            api_url: DeepSeek API endpoint URL
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")

        self.api_url = api_url
        self.model = model
        # Initialize OpenAI async client with DeepSeek API endpoint
        # OpenAI SDK expects base URL without /v1/chat/completions
        # Extract base URL: https://api.deepseek.com/v1/chat/completions -> https://api.deepseek.com
        if "/v1" in api_url:
            base_url = api_url.split("/v1")[0]
        else:
            # If no /v1 in URL, assume it's already a base URL
            base_url = api_url.rstrip("/")
        
        # Configure httpx client - always bypass proxy for better reliability
        # Create custom transport without proxy support
        logger.info("Configuring direct connection (bypassing proxy)")
        
        # Clear any proxy environment variables that httpx might pick up
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        original_proxies = {}
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]
                del os.environ[var]
        
        try:
            # Create client with no proxy
            http_client = httpx.AsyncClient(
                timeout=60.0,
                trust_env=False,  # Don't use environment proxy settings
                proxies=None,  # Explicitly no proxy
            )
            
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=base_url,
                http_client=http_client,
            )
        finally:
            # Restore proxy environment variables
            for var, value in original_proxies.items():
                os.environ[var] = value

    def _build_prompt(self, question: str, chunks: List[dict]) -> str:
        """
        Build prompt with question and retrieved chunks.

        Args:
            question: User's question
            chunks: List of retrieved chunks with metadata

        Returns:
            Formatted prompt string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page_num = chunk.get("page_number", "?")
            text = chunk.get("text", "")
            context_parts.append(f"[Chunk {i}, Page {page_num}]:\n{text}\n")

        context = "\n".join(context_parts)

        prompt = f"""You are a document Q&A assistant. You MUST answer questions ONLY using information from the provided document context. 

CRITICAL RULES:
1. You MUST NOT use any knowledge outside of the provided document context
2. You MUST NOT make up information or use general knowledge
3. If the answer cannot be found in the provided context, you MUST respond with: "I'm sorry, but I cannot answer this question based on the uploaded document. Please ask a question related to the content in the document."
4. Your answer must be exactly 3 paragraphs. Each paragraph should be concise and informative.
5. Do NOT include page numbers in your answer. The page numbers are already displayed separately in the source references.

Context from document:
{context}

Question: {question}

Answer (exactly 3 paragraphs, or the "cannot answer" message if information is not in the context):"""

        return prompt

    async def generate_answer(
        self, question: str, chunks: List[dict]
    ) -> Dict[str, any]:
        """
        Generate answer using DeepSeek API.

        Args:
            question: User's question
            chunks: Retrieved relevant chunks

        Returns:
            Dictionary with answer, token_usage, and response_time_ms
        """
        start_time = time.time()

        prompt = self._build_prompt(question, chunks)

        try:
            # Use OpenAI SDK to call DeepSeek API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a document Q&A assistant. You MUST answer questions ONLY using information from the provided document context. You MUST NOT use any external knowledge or make up information. If the answer cannot be found in the document, you must say so clearly.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,  # Limit to ensure 3 concise paragraphs
            )

            answer = response.choices[0].message.content
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            response_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"LLM response generated",
                extra={
                    "token_usage": token_usage,
                    "response_time_ms": response_time_ms,
                    "answer_length": len(answer),
                },
            )

            return {
                "answer": answer,
                "token_usage": token_usage,
                "response_time_ms": response_time_ms,
            }

        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to generate answer: {str(e)}")

    async def validate_answer_relevance(
        self, question: str, answer: str, chunks: List[dict]
    ) -> bool:
        """
        Validate that the generated answer is actually based on the retrieved chunks.
        
        Args:
            question: User's question
            answer: Generated answer
            chunks: Retrieved chunks used for answer generation
            
        Returns:
            True if answer is based on chunks, False otherwise
        """
        # Extract text from chunks
        chunk_texts = [chunk.get("text", "") for chunk in chunks]
        context = "\n\n".join(chunk_texts)
        
        # Check if answer contains the "cannot answer" message
        cannot_answer_phrases = [
            "cannot answer",
            "cannot be found",
            "not in the context",
            "not available in",
            "not provided",
            "please ask a question related",
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in cannot_answer_phrases):
            return False
        
        # Use LLM to validate if answer is based on context
        validation_prompt = f"""You are a validator. Determine if the answer is based ONLY on the provided document context.

Question: {question}

Document Context:
{context}

Generated Answer:
{answer}

Instructions:
- If the answer can be supported by information in the document context, respond with "YES"
- If the answer uses information NOT in the document context or makes up information, respond with "NO"
- If the answer says it cannot answer, respond with "NO"
- Be strict: only "YES" if the answer is clearly derived from the context

Respond with only "YES" or "NO":"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict validator. Respond with only YES or NO.",
                    },
                    {"role": "user", "content": validation_prompt},
                ],
                temperature=0.1,  # Low temperature for consistent validation
                max_tokens=10,
            )

            validation_result = response.choices[0].message.content.strip().upper()
            is_valid = validation_result == "YES"
            
            logger.info(
                f"Answer validation result: {validation_result}",
                extra={"question": question[:100], "is_valid": is_valid},
            )
            
            return is_valid

        except Exception as e:
            logger.warning(f"Error validating answer relevance: {str(e)}")
            # If validation fails, be conservative and allow the answer
            # But check for obvious "cannot answer" messages
            return not any(phrase in answer_lower for phrase in cannot_answer_phrases)

    async def close(self):
        """Close HTTP client."""
        # Close the async OpenAI client
        await self.client.close()

