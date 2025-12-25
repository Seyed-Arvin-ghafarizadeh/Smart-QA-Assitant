"""Dual memory system: Redis cache (short-term) + Vector store (archival)."""
import hashlib
import json
from typing import Dict, List, Optional

try:
    import redis.asyncio as redis
    from redis.exceptions import ConnectionError, RedisError
except ImportError:
    redis = None
    ConnectionError = Exception
    RedisError = Exception

from app.agent.tools import VectorTool
from app.utils.logger import logger


class MemoryManager:
    """Manages dual memory: Redis cache (short-term) and Vector store (archival)."""

    def __init__(
        self,
        vector_tool: VectorTool,
        redis_url: str = "redis://localhost:6379/0",
        cache_ttl_seconds: int = 3600,
        enable_cache: bool = True,
    ):
        """
        Initialize memory manager.

        Args:
            vector_tool: Vector tool for archival memory
            redis_url: Redis connection URL
            cache_ttl_seconds: TTL for cache entries (default: 1 hour)
            enable_cache: Enable/disable Redis cache
        """
        self.vector_tool = vector_tool
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl_seconds
        self.enable_cache = enable_cache
        self.redis_client: Optional[redis.Redis] = None
        self._cache_available = False

        if enable_cache:
            self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection."""
        if redis is None:
            logger.warning("Redis not installed. Install with: pip install redis. Continuing without cache.")
            self._cache_available = False
            self.redis_client = None
            return
        
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._cache_available = True
            logger.info(f"Redis cache initialized: {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis cache unavailable: {str(e)}. Continuing without cache.")
            self._cache_available = False
            self.redis_client = None

    def _hash_question(self, question: str) -> str:
        """Generate hash for question."""
        return hashlib.sha256(question.encode("utf-8")).hexdigest()

    def _get_cache_key(self, question: str) -> str:
        """Get Redis cache key for question."""
        question_hash = self._hash_question(question)
        return f"qa:{question_hash}"

    async def get_from_cache(self, question: str) -> Optional[Dict]:
        """
        Get answer from short-term cache (Redis).

        Args:
            question: User's question

        Returns:
            Cached answer dict if found, None otherwise
        """
        if not self.enable_cache or not self._cache_available or not self.redis_client:
            return None

        try:
            cache_key = self._get_cache_key(question)
            cached_data = await self.redis_client.get(cache_key)

            if cached_data:
                result = json.loads(cached_data)
                logger.info(f"Cache hit for question: {question[:50]}...")
                return result

            return None

        except (ConnectionError, RedisError) as e:
            logger.warning(f"Redis cache error: {str(e)}")
            self._cache_available = False
            return None
        except Exception as e:
            logger.warning(f"Error reading from cache: {str(e)}")
            return None

    async def store_in_cache(
        self, question: str, answer_data: Dict, ttl: Optional[int] = None
    ) -> bool:
        """
        Store answer in short-term cache (Redis).

        Args:
            question: User's question
            answer_data: Answer data to cache
            ttl: Optional TTL override (uses default if None)

        Returns:
            True if stored successfully, False otherwise
        """
        if not self.enable_cache or not self._cache_available or not self.redis_client:
            return False

        try:
            cache_key = self._get_cache_key(question)
            ttl_to_use = ttl if ttl is not None else self.cache_ttl

            await self.redis_client.setex(
                cache_key, ttl_to_use, json.dumps(answer_data)
            )

            logger.info(f"Cached answer for question: {question[:50]}...")
            return True

        except (ConnectionError, RedisError) as e:
            logger.warning(f"Redis cache error: {str(e)}")
            self._cache_available = False
            return False
        except Exception as e:
            logger.warning(f"Error writing to cache: {str(e)}")
            return False

    async def search_archival(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[dict]:
        """
        Search archival memory (vector store).

        Args:
            query_embedding: Query embedding vector
            top_k: Number of chunks to retrieve

        Returns:
            List of similar chunks from archival memory
        """
        return self.vector_tool.search(query_embedding, top_k=top_k)

    async def store_archival(self, chunks: List, embeddings: List[List[float]]) -> None:
        """
        Store documents in archival memory (vector store).

        Args:
            chunks: List of chunks to store
            embeddings: List of embedding vectors
        """
        # This is handled by VectorStore directly during document upload
        # This method is here for API consistency
        pass

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("Redis cache connection closed")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {str(e)}")

