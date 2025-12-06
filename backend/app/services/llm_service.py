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

        prompt = f"""You are a helpful assistant that answers questions based on the provided document context. Use only the information from the context to answer the question. If the answer cannot be found in the context, say so clearly.

IMPORTANT: Your answer must be exactly 3 paragraphs. Each paragraph should be concise and informative.

When citing information, reference the page number (e.g., "According to page X..." or "[Page X]").

Context from document:
{context}

Question: {question}

Answer (exactly 3 paragraphs):"""

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
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on document context."},
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

    async def close(self):
        """Close HTTP client."""
        # Close the async OpenAI client
        await self.client.close()

