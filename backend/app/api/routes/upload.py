"""Upload endpoint for document processing."""
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends

from app.api.schemas import UploadResponse
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.document_upload_service import DocumentUploadService
from app.utils.logger import logger
from app.exceptions import DocumentProcessingError


router = APIRouter()


def get_document_processor() -> DocumentProcessor:
    """Get document processor service."""
    from app.main import document_processor
    if document_processor is None:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    return document_processor


def get_embedding_service() -> EmbeddingService:
    """Get embedding service from main app."""
    from app.main import embedding_service
    if embedding_service is None:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    return embedding_service


def get_vector_store() -> VectorStore:
    """Get vector store from main app."""
    from app.main import vector_store
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return vector_store


def get_app_settings():
    """Get application settings from main app."""
    from app.main import settings
    if settings is None:
        raise HTTPException(status_code=503, detail="Settings not initialized")
    return settings


def get_upload_service(
    document_processor: DocumentProcessor = Depends(get_document_processor),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    app_settings=Depends(get_app_settings),
) -> DocumentUploadService:
    """Get document upload service with dependencies."""
    return DocumentUploadService(
        document_processor=document_processor,
        embedding_service=embedding_service,
        vector_store=vector_store,
        settings=app_settings,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    upload_service: DocumentUploadService = Depends(get_upload_service),
):
    """
    Upload and process a document (PDF, DOCX, or TXT).

    Args:
        file: Document file to upload and process
        upload_service: Document upload service instance

    Returns:
        UploadResponse with document ID, metadata, and statistics
    """
    try:
        # Read file content
        file_content = await file.read()

        # Process document using the upload service
        result = upload_service.upload_and_process_document(file_content, file.filename)

        return UploadResponse(**result)

    except DocumentProcessingError as e:
        # Convert document processing errors to appropriate HTTP responses
        from app.exceptions import (
            ValidationError, FileTypeNotSupportedError, FileSizeExceededError,
            DocumentCorruptedError, DocumentEmptyError, PageLimitExceededError,
            ChunkLimitExceededError, ExtractionError, EmbeddingError, StorageError,
            ServiceUnavailableError
        )

        if isinstance(e, (ValidationError, FileTypeNotSupportedError)):
            raise HTTPException(status_code=400, detail=str(e))
        elif isinstance(e, (FileSizeExceededError, PageLimitExceededError, ChunkLimitExceededError)):
            raise HTTPException(status_code=400, detail=str(e))
        elif isinstance(e, (DocumentCorruptedError, DocumentEmptyError, ExtractionError)):
            raise HTTPException(status_code=400, detail=str(e))
        elif isinstance(e, (EmbeddingError, StorageError)):
            raise HTTPException(status_code=500, detail=str(e))
        elif isinstance(e, ServiceUnavailableError):
            raise HTTPException(status_code=503, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        )

