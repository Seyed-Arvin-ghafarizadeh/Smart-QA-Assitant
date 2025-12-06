"""FastAPI application entry point."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import json

from app.api.routes import ask, metrics, upload
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStore
from app.utils.logger import logger
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    deepseek_api_key: str = ""
    deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions"
    chroma_db_path: str = "./chroma_db"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    top_k_chunks: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    class Config:
        # Look for .env in both backend/ and parent directory
        env_file = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = "ignore"


# Global services (initialized in lifespan)
embedding_service: EmbeddingService = None
vector_store: VectorStore = None
llm_service: LLMService = None
settings: Settings = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global embedding_service, vector_store, llm_service, settings

    # Startup
    logger.info("Starting Smart Document QA Assistant")
    settings = Settings()

    # Initialize services
    embedding_service = EmbeddingService(model_name=settings.embedding_model)
    vector_store = VectorStore(db_path=settings.chroma_db_path)
    llm_service = LLMService(
        api_key=settings.deepseek_api_key, api_url=settings.deepseek_api_url
    )

    logger.info("All services initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Smart Document QA Assistant")
    if llm_service:
        await llm_service.close()


# Create FastAPI app
app = FastAPI(
    title="Smart Document QA Assistant",
    description="RAG-based document question answering system",
    version="1.0.0",
    lifespan=lifespan,
)


# Exception handler for JSON parsing errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors with better error messages.
    
    Specifically handles JSON decode errors from invalid control characters.
    """
    errors = exc.errors()
    
    # Check if it's a JSON decode error
    for error in errors:
        if error.get("type") == "json_invalid":
            error_msg = error.get("msg", "Invalid JSON")
            ctx = error.get("ctx", {})
            if "Invalid control character" in str(ctx.get("error", "")):
                return JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    content={
                        "detail": "Invalid JSON: Control characters detected in request body. "
                                 "Please ensure your question doesn't contain special control characters.",
                        "error": "json_parse_error",
                        "hint": "Remove any special characters from your question or use proper JSON encoding."
                    }
                )
    
    # Return standard validation error response
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors}
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Smart Document QA Assistant"}


# Prometheus metrics endpoint
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Include routers
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(ask.router, prefix="/api", tags=["ask"])
app.include_router(metrics.router, prefix="/api", tags=["metrics"])


# Services are accessed via dependency injection in route modules

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host if settings else "0.0.0.0", port=settings.api_port if settings else 8000)

