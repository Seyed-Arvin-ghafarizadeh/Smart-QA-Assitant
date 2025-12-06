"""Upload endpoint for document processing."""
import os
import uuid
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends

from app.api.schemas import UploadResponse
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.utils.logger import logger
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Upload settings."""

    chunk_size: int = 400
    chunk_overlap: int = 50
    upload_dir: str = "./uploads"

    class Config:
        env_file = ".env"
        case_sensitive = False


router = APIRouter()
settings = Settings()

# Create upload directory
os.makedirs(settings.upload_dir, exist_ok=True)


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


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    embedding_svc: EmbeddingService = Depends(get_embedding_service),
    vector_svc: VectorStore = Depends(get_vector_store),
):
    """
    Upload and process a PDF document.

    Args:
        file: PDF file to upload
        embedding_svc: Embedding service instance
        vector_svc: Vector store instance

    Returns:
        UploadResponse with document ID and metadata
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate unique document ID
    document_id = str(uuid.uuid4())

    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".pdf", dir=settings.upload_dir) as tmp_file:
        try:
            # Read file content
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

            # Process document
            processor = DocumentProcessor(
                chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
            )
            document = processor.process_document(tmp_file_path, document_id, file.filename)

            # Generate embeddings
            chunk_texts = [chunk.text for chunk in document.chunks]
            embeddings = embedding_svc.generate_embeddings(chunk_texts)

            # Store in vector database
            vector_svc.store_document(document.chunks, embeddings)

            logger.info(
                f"Document uploaded successfully",
                extra={
                    "document_id": document_id,
                    "doc_filename": file.filename,
                    "total_chunks": document.total_chunks,
                },
            )

            return UploadResponse(
                document_id=document_id,
                filename=file.filename,
                total_chunks=document.total_chunks,
                total_pages=document.total_pages,
            )

        except ValueError as e:
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error uploading document: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to process document")
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass

