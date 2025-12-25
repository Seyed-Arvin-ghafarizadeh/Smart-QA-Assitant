"""FastAPI application entry point."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import json
import os

from app.api.routes import ask, metrics, upload
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.ocr_service import EasyOCRService
from app.agent.agent import DocumentQAAgent
from app.agent.tools import LLMTool, EmbeddingTool, VectorTool
from app.agent.memory import MemoryManager
from app.agent.validators import QuestionValidator, AnswerValidator, OutputValidator
from app.utils.logger import logger
from app.utils.tracer import initialize_tracing, shutdown_tracing
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    deepseek_api_key: str = ""
    deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions"
    qdrant_db_path: str = "./qdrant_db"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    top_k_chunks: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Document upload limits
    max_file_size_mb: int = 50  # Maximum file size in MB
    max_pages: int = 1000  # Maximum number of pages
    max_characters: int = 2000000  # Maximum characters (2M)
    max_words: int = 300000  # Maximum words (300k)
    max_chunks: int = 5000  # Maximum chunks
    max_sentences: int = 10000  # Maximum sentences (10k)

    # Processing configuration - optimized for performance
    qdrant_batch_size: int = 500  # Qdrant batch size for storing chunks (increased from 100)
    embedding_batch_size: int = 256  # Batch size for embedding generation (increased for better RAM utilization)

    # Document chunking configuration - adjusted for better retrieval
    chunk_size: int = 500  # Smaller chunks for better granularity (was 1000)
    chunk_overlap: int = 100  # Overlap between chunks

    # CPU and resource configuration
    cpu_cores: int = 0  # Number of CPU cores to use (0 = use all available)
    max_workers: int = 0  # Max workers for parallel processing (0 = auto: cores * 2, max 16)

    # OCR configuration (EasyOCR)
    ocr_enabled: bool = True  # Enable/disable OCR fallback
    ocr_languages: str = "en"  # Comma-separated language codes (e.g., "en,fr,es" for English, French, Spanish)
    ocr_gpu: bool = False  # Use GPU for OCR (requires CUDA)
    ocr_dpi: int = 300  # DPI for OCR image conversion (higher = better quality but slower)
    ocr_text_threshold: int = 50  # Minimum characters before using OCR fallback

    # Similarity threshold configuration (for semantic search)
    similarity_max_threshold: float = 0.20  # Maximum similarity threshold (lowered for better recall with query expansion)
    similarity_avg_threshold: float = 0.15  # Average similarity threshold (lowered for better recall with query expansion)
    similarity_min_score: float = 0.15  # Minimum score for "high similarity" chunks
    similarity_high_count_min: int = 1  # Minimum number of chunks with high similarity
    
    # Query expansion configuration
    enable_query_expansion: bool = True  # Enable query expansion using LLM to improve retrieval

    # Performance optimization: Skip expensive LLM validation checks
    skip_relevance_check: bool = True  # Skip LLM-based relevance check (saves 1 API call, faster responses)
    skip_answer_validation: bool = True  # Skip LLM-based answer validation (saves 1 API call, faster responses)

    # OpenTelemetry tracing configuration
    tracing_enabled: bool = True  # Enable/disable tracing
    otlp_endpoint: str = ""  # OTLP endpoint URL (empty = use console exporter)

    # Redis cache configuration
    redis_url: str = "redis://localhost:6379/0"  # Redis connection URL
    cache_ttl_seconds: int = 3600  # Cache TTL in seconds (default: 1 hour)
    enable_cache: bool = True  # Enable/disable Redis cache

    class Config:
        # Look for .env in both backend/ and parent directory
        env_file = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = "ignore"


# Global services (initialized in lifespan)
document_processor: DocumentProcessor = None
embedding_service: EmbeddingService = None
vector_store: VectorStore = None
agent: DocumentQAAgent = None
settings: Settings = None
tracer_provider = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global document_processor, embedding_service, vector_store, agent, settings, tracer_provider

    # Startup
    logger.info("Starting Smart Document QA Assistant")
    settings = Settings()

    # Initialize tracing before services
    tracer_provider = initialize_tracing(
        service_name="smart-document-qa",
        service_version="1.0.0",
        otlp_endpoint=settings.otlp_endpoint if settings.otlp_endpoint else None,
        tracing_enabled=settings.tracing_enabled,
    )

    # Initialize OCR service (if enabled)
    ocr_service = None
    if settings.ocr_enabled:
        # Parse language codes from comma-separated string
        ocr_languages = [lang.strip() for lang in settings.ocr_languages.split(",") if lang.strip()]
        if not ocr_languages:
            ocr_languages = ["en"]  # Default to English
        
        try:
            ocr_service = EasyOCRService(languages=ocr_languages, gpu=settings.ocr_gpu)
            logger.info(f"OCR service initialized with languages: {ocr_languages}")
        except Exception as e:
            logger.warning(f"Failed to initialize OCR service: {str(e)}. OCR will be disabled.")
            ocr_service = None

    # Initialize services (model loaded lazily on first use)
    document_processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_workers=settings.max_workers if settings.max_workers > 0 else 0,
        ocr_service=ocr_service,
        ocr_enabled=settings.ocr_enabled,
        ocr_text_threshold=settings.ocr_text_threshold,
        ocr_dpi=settings.ocr_dpi,
    )
    embedding_service = EmbeddingService(
        model_name=settings.embedding_model,
        cpu_cores=settings.cpu_cores,
        batch_size=settings.embedding_batch_size,
        use_parallel=True,  # Enable parallel batch processing
        max_parallel_workers=settings.max_workers if settings.max_workers > 0 else 0  # Auto-detect if 0
    )
    vector_store = VectorStore(
        db_path=settings.qdrant_db_path,
        batch_size=settings.qdrant_batch_size
    )

    # Initialize agent components
    llm_tool = LLMTool(
        api_key=settings.deepseek_api_key,
        api_url=settings.deepseek_api_url
    )
    embedding_tool = EmbeddingTool(embedding_service)
    vector_tool = VectorTool(vector_store)
    
    memory_manager = MemoryManager(
        vector_tool=vector_tool,
        redis_url=settings.redis_url,
        cache_ttl_seconds=settings.cache_ttl_seconds,
        enable_cache=settings.enable_cache,
    )
    
    question_validator = QuestionValidator()
    answer_validator = AnswerValidator()
    output_validator = OutputValidator()

    # Initialize agent
    agent = DocumentQAAgent(
        llm_tool=llm_tool,
        embedding_tool=embedding_tool,
        vector_tool=vector_tool,
        memory_manager=memory_manager,
        question_validator=question_validator,
        answer_validator=answer_validator,
        output_validator=output_validator,
        similarity_max_threshold=settings.similarity_max_threshold,
        similarity_avg_threshold=settings.similarity_avg_threshold,
        similarity_min_score=settings.similarity_min_score,
        similarity_high_count_min=settings.similarity_high_count_min,
        skip_relevance_check=settings.skip_relevance_check,
        skip_answer_validation=settings.skip_answer_validation,
        enable_query_expansion=settings.enable_query_expansion,
    )

    logger.info("All services and agent initialized successfully (embedding model will load on first use)")
    if settings.skip_relevance_check or settings.skip_answer_validation:
        logger.info(
            f"Performance optimizations enabled: "
            f"skip_relevance_check={settings.skip_relevance_check}, "
            f"skip_answer_validation={settings.skip_answer_validation} "
            f"(reduces LLM API calls from 3 to 1 per question)"
        )
    if settings.enable_query_expansion:
        logger.info(
            f"Query expansion enabled: Will expand user queries using LLM to improve retrieval "
            f"(similarity thresholds: max={settings.similarity_max_threshold}, "
            f"avg={settings.similarity_avg_threshold}, min={settings.similarity_min_score})"
        )

    yield

    # Shutdown
    logger.info("Shutting down Smart Document QA Assistant")
    if agent:
        await agent.close()
    if tracer_provider:
        shutdown_tracing(tracer_provider)


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

# Serve static frontend files (if available)
# Try multiple paths: Docker container path, relative paths for local development
static_paths = [
    "/app/frontend/static",  # Docker container path
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "static"),  # Local dev path
    os.path.join(os.getcwd(), "frontend", "static"),  # Current working directory
]

static_dir = None
for path in static_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path) and os.path.isdir(abs_path):
        static_dir = abs_path
        break

if static_dir:
    try:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
        logger.info(f"Serving static frontend from {static_dir}")
    except Exception as e:
        logger.warning(f"Failed to mount static files: {e}")
else:
    logger.info("Static frontend directory not found, skipping static file serving")


# Services are accessed via dependency injection in route modules

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host if settings else "0.0.0.0", port=settings.api_port if settings else 8000)

