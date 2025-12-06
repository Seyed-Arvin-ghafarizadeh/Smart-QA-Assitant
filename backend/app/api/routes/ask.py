"""Ask endpoint for question answering."""
from fastapi import APIRouter, HTTPException, Depends

from app.api.schemas import AskRequest, AskResponse
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStore
from app.utils.logger import logger

router = APIRouter()


def get_embedding_service() -> EmbeddingService:
    """Get embedding service from main app."""
    from app.main import embedding_service
    if embedding_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return embedding_service


def get_vector_store() -> VectorStore:
    """Get vector store from main app."""
    from app.main import vector_store
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return vector_store


def get_llm_service() -> LLMService:
    """Get LLM service from main app."""
    from app.main import llm_service
    if llm_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return llm_service


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    embedding_svc: EmbeddingService = Depends(get_embedding_service),
    vector_svc: VectorStore = Depends(get_vector_store),
    llm_svc: LLMService = Depends(get_llm_service),
):
    """
    Answer a question about uploaded documents.
    
    Searches across all uploaded documents and returns a concise 3-paragraph answer.

    Args:
        request: AskRequest with question
        embedding_svc: Embedding service instance
        vector_svc: Vector store instance
        llm_svc: LLM service instance

    Returns:
        AskResponse with answer (3 paragraphs), relevant chunks, and metadata
    """
    try:
        # Create retrieval service instance
        retrieval_svc = RetrievalService(embedding_svc, vector_svc, llm_svc)

        # Perform RAG pipeline (searches across all documents)
        result = await retrieval_svc.retrieve_and_answer(
            question=request.question,
            top_k=5,  # Fixed to 5 chunks for optimal performance
        )

        return AskResponse(
            answer=result["answer"],
            relevant_chunks=result["relevant_chunks"],
            confidence=result.get("confidence"),
            token_usage=result.get("token_usage"),
            response_time_ms=result.get("response_time_ms"),
        )

    except ValueError as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error answering question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate answer")

