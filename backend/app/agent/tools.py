"""Tools for LLM, embedding, and vector operations."""
import os
import time
from typing import Dict, List, Optional
import httpx
from openai import AsyncOpenAI

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.agent.prompts import AnswerPrompt, ValidationPrompt
from app.utils.logger import logger


class LLMTool:
    """Tool for LLM operations using OpenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.deepseek.com/v1/chat/completions",
        model: str = "deepseek-chat",
    ):
        """
        Initialize LLM tool.

        Args:
            api_key: API key (from env if not provided)
            api_url: API endpoint URL
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")

        self.api_url = api_url
        self.model = model

        # Extract base URL
        if "/v1" in api_url:
            base_url = api_url.split("/v1")[0]
        else:
            base_url = api_url.rstrip("/")

        # Configure httpx client - bypass proxy
        logger.info("Configuring direct connection (bypassing proxy)")
        proxy_vars = [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ]
        original_proxies = {}
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]
                del os.environ[var]

        try:
            http_client = httpx.AsyncClient(
                timeout=60.0,
                trust_env=False,
                proxies=None,
            )
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=base_url,
                http_client=http_client,
            )
        finally:
            for var, value in original_proxies.items():
                os.environ[var] = value

    async def generate_answer(
        self, question: str, chunks: List[dict], temperature: float = 0.7, max_tokens: int = 500
    ) -> Dict[str, any]:
        """
        Generate answer using LLM.

        Args:
            question: User's question
            chunks: Retrieved relevant chunks
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response

        Returns:
            Dictionary with answer, token_usage, and response_time_ms
        """
        start_time = time.time()
        prompt = AnswerPrompt.build(question, chunks)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": AnswerPrompt.SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = response.choices[0].message.content
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            response_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "LLM response generated",
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
            logger.error(f"Error calling LLM API: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to generate answer: {str(e)}")

    async def validate_answer(
        self, question: str, answer: str, chunks: List[dict]
    ) -> bool:
        """
        Validate answer relevance using LLM.

        Args:
            question: User's question
            answer: Generated answer
            chunks: Retrieved chunks used for answer generation

        Returns:
            True if answer is based on chunks, False otherwise
        """
        chunk_texts = [chunk.get("text", "") for chunk in chunks]
        context = "\n\n".join(chunk_texts)

        # Check for "cannot answer" phrases
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

        prompt = ValidationPrompt.build(question, answer, context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ValidationPrompt.SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
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
            return not any(phrase in answer_lower for phrase in cannot_answer_phrases)

    async def analyze_relevance(
        self, question: str, document_summary: Optional[str] = None
    ) -> Dict:
        """
        Analyze question relevance using LLM.

        Args:
            question: User's question
            document_summary: Optional summary of document content

        Returns:
            Dictionary with is_relevant, sentiment, confidence, reason
        """
        from app.agent.prompts import RelevancePrompt

        start_time = time.time()
        prompt = RelevancePrompt.build(question, document_summary)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RelevancePrompt.SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=150,
            )

            result_text = response.choices[0].message.content.strip()

            # Parse response - default to True (relevant) for safety
            is_relevant = True
            sentiment = "neutral"
            confidence = 0.5
            reason = "Analysis completed"

            lines = result_text.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("RELEVANT:"):
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
                "Question relevance analysis completed",
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
            return {
                "is_relevant": True,
                "sentiment": "neutral",
                "confidence": 0.5,
                "reason": f"Analysis error: {str(e)}",
                "response_time_ms": (time.time() - start_time) * 1000,
            }

    async def expand_query(self, question: str) -> str:
        """
        Expand user query using LLM to improve semantic search retrieval.
        Uses "hallucination" approach to generate multiple related queries and concepts.
        
        Args:
            question: Original user question
            
        Returns:
            Expanded query string with related terms and concepts
        """
        start_time = time.time()
        
        expansion_prompt = f"""You are a query expansion assistant. Your job is to expand a user's question into a more detailed, comprehensive query that will help find relevant information in documents.

Original Question: {question}

Your task:
1. Identify the main topic, person, concept, or entity mentioned in the question
2. Generate related terms, synonyms, alternative phrasings, and related concepts
3. Think about what information someone asking this question might be looking for
4. Include variations of how this topic might be described in documents
5. Make the expanded query comprehensive but focused

IMPORTANT: 
- Include the original question terms
- Add related terms, synonyms, and alternative phrasings
- Think about what context or background information might be relevant
- Consider different ways the topic might be mentioned in documents
- Make it longer and more detailed than the original

Respond with ONLY the expanded query text (no explanations, no labels, just the expanded query):"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful query expansion assistant. Expand queries to improve document retrieval."
                    },
                    {"role": "user", "content": expansion_prompt},
                ],
                temperature=0.7,  # Higher temperature for more creative expansion
                max_tokens=200,
            )

            expanded_query = response.choices[0].message.content.strip()
            
            # Combine original and expanded query for better results
            final_query = f"{question} {expanded_query}"
            
            response_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "Query expansion completed",
                extra={
                    "original_question": question[:100],
                    "expanded_query": final_query[:200],
                    "response_time_ms": response_time_ms,
                },
            )

            return final_query

        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}", exc_info=True)
            # Fallback to original question if expansion fails
            return question

    async def close(self):
        """Close HTTP client."""
        await self.client.close()


class EmbeddingTool:
    """Tool for embedding generation."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
    ):
        """
        Initialize embedding tool.

        Args:
            embedding_service: Embedding service instance
        """
        self.embedding_service = embedding_service

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.embedding_service.generate_embedding(text)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self.embedding_service.generate_embeddings(texts)


class VectorTool:
    """Tool for vector store operations (archival memory)."""

    def __init__(self, vector_store: VectorStore):
        """
        Initialize vector tool.

        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[dict]:
        """
        Search archival memory for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of chunks to retrieve

        Returns:
            List of similar chunks with metadata
        """
        return self.vector_store.retrieve_similar_across_all(query_embedding, top_k=top_k)

