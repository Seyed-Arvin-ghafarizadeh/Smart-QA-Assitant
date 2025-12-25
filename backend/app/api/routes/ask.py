"""Ask endpoint for question answering."""
from fastapi import APIRouter, HTTPException, Depends

from app.api.schemas import AskRequest, AskResponse
from app.agent.agent import DocumentQAAgent
from app.utils.logger import logger

router = APIRouter()


def get_agent() -> DocumentQAAgent:
    """Get agent from main app."""
    from app.main import agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    agent: DocumentQAAgent = Depends(get_agent),
):
    """
    Answer a question about uploaded documents.
    
    Uses the unified agent with dual memory system (Redis cache + vector store).
    Searches across all uploaded documents and returns a concise 3-paragraph answer.

    Args:
        request: AskRequest with question
        agent: Document Q&A agent instance

    Returns:
        AskResponse with answer (3 paragraphs), relevant chunks, and metadata
    """
    try:
        # Use agent to answer question
        result = await agent.ask(
            question=request.question,
            top_k=5,  # Fixed to 5 chunks for optimal performance
        )

        return AskResponse(
            answer=result["answer"],
            relevant_chunks=result["relevant_chunks"],
            confidence=result.get("confidence"),
            token_usage=result.get("token_usage"),
            response_time_ms=result.get("response_time_ms"),
            sentiment=result.get("sentiment"),
            is_relevant=result.get("is_relevant", True),
            similarity_metrics=result.get("similarity_metrics"),
        )

    except ValueError as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error answering question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate answer")
