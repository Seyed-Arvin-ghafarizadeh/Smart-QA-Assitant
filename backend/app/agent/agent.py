"""Main agent class orchestrating the RAG pipeline with dual memory."""
import time
from typing import Dict, Optional

from app.agent.tools import LLMTool, EmbeddingTool, VectorTool
from app.agent.memory import MemoryManager
from app.agent.validators import QuestionValidator, AnswerValidator, OutputValidator
from app.utils.logger import logger


class DocumentQAAgent:
    """Unified agent for document Q&A with dual memory system."""

    def __init__(
        self,
        llm_tool: LLMTool,
        embedding_tool: EmbeddingTool,
        vector_tool: VectorTool,
        memory_manager: MemoryManager,
        question_validator: QuestionValidator,
        answer_validator: AnswerValidator,
        output_validator: OutputValidator,
        similarity_max_threshold: float = 0.40,
        similarity_avg_threshold: float = 0.35,
        similarity_min_score: float = 0.35,
        similarity_high_count_min: int = 1,
        skip_relevance_check: bool = True,
        skip_answer_validation: bool = True,
        enable_query_expansion: bool = True,
    ):
        """
        Initialize document Q&A agent.

        Args:
            llm_tool: LLM tool for generation and analysis
            embedding_tool: Embedding tool for query embeddings
            vector_tool: Vector tool for archival memory
            memory_manager: Memory manager for dual memory
            question_validator: Question validator
            answer_validator: Answer validator
            output_validator: Output validator
            similarity_max_threshold: Maximum similarity threshold
            similarity_avg_threshold: Average similarity threshold
            similarity_min_score: Minimum score for high similarity
            similarity_high_count_min: Minimum chunks with high similarity
            skip_relevance_check: Skip expensive LLM relevance check (faster, less strict)
            skip_answer_validation: Skip expensive LLM answer validation (faster, less strict)
            enable_query_expansion: Enable query expansion using LLM to improve retrieval
        """
        self.llm_tool = llm_tool
        self.embedding_tool = embedding_tool
        self.vector_tool = vector_tool
        self.memory = memory_manager
        self.question_validator = question_validator
        self.answer_validator = answer_validator
        self.output_validator = output_validator
        self.similarity_max_threshold = similarity_max_threshold
        self.similarity_avg_threshold = similarity_avg_threshold
        self.similarity_min_score = similarity_min_score
        self.similarity_high_count_min = similarity_high_count_min
        self.skip_relevance_check = skip_relevance_check
        self.skip_answer_validation = skip_answer_validation
        self.enable_query_expansion = enable_query_expansion

    async def ask(self, question: str, session_id: Optional[str] = None, top_k: int = 5) -> Dict:
        """
        Answer a question using the RAG pipeline with dual memory.

        Args:
            question: User's question
            session_id: Optional session ID for conversation context
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with answer, chunks, and metadata
        """
        start_time = time.time()

        # Step 1: Check short-term cache
        cached_result = await self.memory.get_from_cache(question)
        if cached_result:
            logger.info("Returning cached answer")
            return cached_result

        # Step 2: Validate question
        sentiment_result = self.question_validator.validate_sentiment(question)
        if not sentiment_result.get("is_appropriate", True):
            total_time = (time.time() - start_time) * 1000
            return {
                "answer": "I'm sorry, but I cannot answer this question. Please ask a professional question related to the content in the uploaded documents.",
                "relevant_chunks": [],
                "confidence": 0.0,
                "token_usage": None,
                "response_time_ms": total_time,
                "cannot_answer": True,
                "sentiment": sentiment_result.get("sentiment", "inappropriate"),
                "is_relevant": False,
            }

        # Step 3: Expand query using LLM (if enabled) to improve retrieval
        search_query = question
        if self.enable_query_expansion:
            logger.info("Expanding query using LLM to improve retrieval")
            search_query = await self.llm_tool.expand_query(question)
            logger.info(f"Original query: {question[:100]}, Expanded query: {search_query[:200]}")

        # Step 4: Generate query embedding (using expanded query if enabled)
        query_embedding = self.embedding_tool.generate_embedding(search_query)

        # Step 5: Search archival memory
        retrieved_chunks = await self.memory.search_archival(query_embedding, top_k=top_k)

        if not retrieved_chunks:
            total_time = (time.time() - start_time) * 1000
            raise ValueError(
                "No matching documents found. This could mean: "
                "(1) No documents have been uploaded, or "
                "(2) The embedding model was changed and old documents need to be re-uploaded."
            )

        # Step 6: Validate similarity thresholds
        similarity_scores = [chunk["similarity_score"] for chunk in retrieved_chunks]
        similarity_check = self.answer_validator.validate_similarity_thresholds(
            similarity_scores,
            self.similarity_max_threshold,
            self.similarity_avg_threshold,
            self.similarity_min_score,
            self.similarity_high_count_min,
        )

        if not similarity_check["passed"]:
            total_time = (time.time() - start_time) * 1000
            return {
                "answer": "I'm sorry, but I cannot answer this question based on the uploaded document. Please ask a question that is more directly related to the content in the document.",
                "relevant_chunks": [
                    {
                        "text": chunk["text"],
                        "page_number": chunk["page_number"],
                        "chapter_number": chunk.get("chapter_number"),
                        "similarity_score": chunk["similarity_score"],
                        "document_id": chunk.get("document_id"),
                    }
                    for chunk in retrieved_chunks
                ],
                "confidence": similarity_check["metrics"].get("avg_similarity", 0.0),
                "token_usage": None,
                "response_time_ms": total_time,
                "cannot_answer": True,
                "sentiment": sentiment_result.get("sentiment", "neutral"),
                "is_relevant": False,
                "similarity_metrics": similarity_check["metrics"],
            }

        # Step 7: LLM-based relevance check (optional, can be skipped for performance)
        if self.skip_relevance_check:
            # Skip expensive LLM relevance check - trust similarity scores
            relevance_analysis = {
                "is_relevant": True,
                "sentiment": sentiment_result.get("sentiment", "neutral"),
                "confidence": similarity_check["metrics"].get("avg_similarity", 0.0),
                "reason": "Skipped (performance optimization)",
            }
        else:
            chunk_texts = [chunk["text"][:200] for chunk in retrieved_chunks[:3]]
            document_summary = "\n".join(chunk_texts)

            relevance_analysis = await self.llm_tool.analyze_relevance(question, document_summary)

            if not relevance_analysis.get("is_relevant", True):
                total_time = (time.time() - start_time) * 1000
                return {
                    "answer": "I'm sorry, but this question does not appear to be related to the content in the uploaded documents. Please ask a question that is directly related to the document content.",
                    "relevant_chunks": [
                        {
                            "text": chunk["text"],
                            "page_number": chunk["page_number"],
                            "chapter_number": chunk.get("chapter_number"),
                            "similarity_score": chunk["similarity_score"],
                            "document_id": chunk.get("document_id"),
                        }
                        for chunk in retrieved_chunks
                    ],
                    "confidence": similarity_check["metrics"].get("avg_similarity", 0.0),
                    "token_usage": None,
                    "response_time_ms": total_time,
                    "cannot_answer": True,
                    "sentiment": relevance_analysis.get("sentiment", "neutral"),
                    "is_relevant": False,
                    "relevance_reason": relevance_analysis.get("reason", ""),
                    "similarity_metrics": similarity_check["metrics"],
                }

        # Step 8: Generate answer using LLM (use original question, not expanded)
        llm_result = await self.llm_tool.generate_answer(question, retrieved_chunks)
        answer = llm_result["answer"]

        # Step 9: Validate answer (optional, can be skipped for performance)
        llm_says_cannot_answer = self.answer_validator.check_cannot_answer(answer)

        if not llm_says_cannot_answer and not self.skip_answer_validation:
            is_answerable = await self.llm_tool.validate_answer(
                question, answer, retrieved_chunks
            )

            if not is_answerable:
                logger.info("Answer validation determined answer is not based on document content")
                answer = "I'm sorry, but I cannot answer this question based on the uploaded document. Please ask a question that is directly related to the content in the document."
                llm_result["answer"] = answer
                llm_says_cannot_answer = True

        total_time = (time.time() - start_time) * 1000

        # Step 10: Format response
        response = {
            "answer": answer,
            "relevant_chunks": [
                {
                    "text": chunk["text"],
                    "page_number": chunk["page_number"],
                    "chapter_number": chunk.get("chapter_number"),
                    "similarity_score": chunk["similarity_score"],
                    "document_id": chunk.get("document_id"),
                }
                for chunk in retrieved_chunks
            ],
            "confidence": similarity_check["metrics"].get("avg_similarity", 0.0),
            "token_usage": llm_result.get("token_usage"),
            "response_time_ms": total_time,
            "sentiment": relevance_analysis.get(
                "sentiment", sentiment_result.get("sentiment", "neutral")
            ),
            "is_relevant": True,
            "similarity_metrics": similarity_check["metrics"],
        }

        if llm_says_cannot_answer:
            response["cannot_answer"] = True
            response["is_relevant"] = False

        # Step 11: Validate output format
        output_validation = self.output_validator.validate_response(response)
        if not output_validation["is_valid"]:
            logger.warning(f"Output validation errors: {output_validation['errors']}")

        # Step 12: Store in cache
        await self.memory.store_in_cache(question, response)

        logger.info(
            "RAG pipeline completed",
            extra={
                "llm_response_time": llm_result.get("response_time_ms"),
                "total_time_ms": total_time,
                "token_usage": llm_result.get("token_usage"),
                "answer_length": len(answer),
                "sentiment": response.get("sentiment"),
                "is_relevant": response.get("is_relevant"),
            },
        )

        return response

    async def close(self):
        """Close agent resources."""
        await self.memory.close()
        await self.llm_tool.close()

