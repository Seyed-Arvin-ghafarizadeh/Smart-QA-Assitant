"""Document upload service for handling file uploads and processing."""
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from tempfile import NamedTemporaryFile

from app.models.document import Chunk
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.validators import validate_document
from app.utils.logger import logger
from app.exceptions import (
    DocumentProcessingError,
    ChunkLimitExceededError,
    EmbeddingError,
    StorageError
)


class DocumentUploadService:
    """Service for handling document uploads and processing."""

    def __init__(
        self,
        document_processor: DocumentProcessor,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        settings: Dict[str, Any],
        upload_dir: str = "./uploads"
    ):
        """
        Initialize upload service.

        Args:
            document_processor: Document processing service
            embedding_service: Embedding generation service
            vector_store: Vector storage service
            settings: Application settings
            upload_dir: Directory for temporary file storage
        """
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.settings = settings
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)

    def upload_and_process_document(
        self,
        file_content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """
        Upload and process a document.

        Args:
            file_content: Raw file content as bytes
            filename: Original filename

        Returns:
            Dict with processing results including document_id, metadata, and statistics

        Raises:
            Various document processing exceptions
        """
        document_id = str(uuid.uuid4())
        tmp_file_path = None

        try:
            # Step 1: Save file temporarily
            tmp_file_path = self._save_temporary_file(file_content, filename)

            # Step 2: Validate document
            validation_result = validate_document(
                tmp_file_path,
                len(file_content),
                self.settings
            )

            # Step 3: Process document
            processing_result = self._process_document(tmp_file_path, document_id, filename)

            # Step 4: Validate chunk limits
            self._validate_chunk_limits(processing_result['chunks'])

            # Step 5: Generate embeddings and store
            storage_result = self._store_document_chunks(
                processing_result['chunks'],
                document_id
            )

            # Step 6: Compile final result
            result = self._compile_upload_result(
                document_id,
                filename,
                validation_result,
                processing_result,
                storage_result
            )

            logger.info(
                f"Document uploaded successfully: {document_id}",
                extra={
                    "document_id": document_id,
                    "uploaded_filename": filename,
                    "total_chunks": result["total_chunks"],
                    "total_pages": result["total_pages"],
                    "processing_time_seconds": result["processing_time_seconds"],
                }
            )

            return result

        except Exception as e:
            logger.error(f"Document upload failed for {filename}: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass

    def _save_temporary_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file to temporary location."""
        file_extension = Path(filename).suffix or '.tmp'

        with NamedTemporaryFile(delete=False, suffix=file_extension, dir=self.upload_dir) as tmp_file:
            tmp_file.write(file_content)
            return tmp_file.name

    def _process_document(
        self,
        file_path: str,
        document_id: str,
        filename: str
    ) -> Dict[str, Any]:
        """Process document and extract chunks."""
        start_time = time.time()

        try:
            chunks = self.document_processor.process_document(file_path, document_id)

            processing_time = time.time() - start_time

            # Calculate statistics
            pages_with_text = len(set(chunk.page_number for chunk in chunks if chunk.text.strip()))
            total_text_chars = sum(len(chunk.text) for chunk in chunks)

            # Log processing details
            file_extension = Path(filename).suffix
            if file_extension == '.pdf':
                logger.info(
                    f"Extraction summary for {filename}: "
                    f"{pages_with_text} pages with text, "
                    f"{total_text_chars:,} characters extracted, "
                    f"{len(chunks):,} chunks created"
                )
            else:
                logger.info(
                    f"Extraction summary for {filename}: "
                    f"{total_text_chars:,} characters extracted, "
                    f"{len(chunks):,} chunks created"
                )

            return {
                'chunks': chunks,
                'processing_time': processing_time,
                'pages_with_text': pages_with_text,
                'total_text_chars': total_text_chars
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Document processing failed after {processing_time:.2f}s: {str(e)}")
            raise DocumentProcessingError(f"Failed to process document: {str(e)}")

    def _validate_chunk_limits(self, chunks: List[Chunk]) -> None:
        """Validate that document doesn't exceed chunk limits."""
        total_chunks = len(chunks)
        if total_chunks >= self.settings.max_chunks:
            # Flush any buffered chunks before raising error
            self.vector_store.flush_all_buffers()
            raise ChunkLimitExceededError(
                f"Document would create more than {self.settings.max_chunks:,} chunks. "
                f"Stopped at {total_chunks:,} chunks."
            )

    def _store_document_chunks(self, chunks: List[Chunk], document_id: str) -> Dict[str, Any]:
        """Generate embeddings and store document chunks."""
        start_time = time.time()
        total_chunks = len(chunks)

        if total_chunks == 0:
            return {'stored_chunks': 0, 'storage_time': 0}

        try:
            # Process in batches for efficiency
            embedding_batch_size = getattr(self.settings, 'embedding_batch_size', 32)
            stored_count = 0

            for i in range(0, total_chunks, embedding_batch_size):
                batch_end = min(i + embedding_batch_size, total_chunks)
                batch_chunks = chunks[i:batch_end]

                # Extract texts for embedding
                batch_texts = [chunk.text for chunk in batch_chunks]

                # Generate embeddings
                embeddings = self.embedding_service.generate_embeddings(batch_texts, embedding_batch_size)

                if len(embeddings) != len(batch_chunks):
                    raise EmbeddingError("Embedding generation failed: mismatched batch sizes")

                # Store chunks with embeddings
                for chunk, embedding in zip(batch_chunks, embeddings):
                    self.vector_store.store_chunk(chunk, embedding)
                    stored_count += 1

            # Flush any remaining buffered chunks
            self.vector_store.flush_all_buffers()

            storage_time = time.time() - start_time

            logger.info(f"Successfully stored {stored_count} chunks for document {document_id}")

            return {
                'stored_chunks': stored_count,
                'storage_time': storage_time
            }

        except Exception as e:
            storage_time = time.time() - start_time
            logger.error(f"Document storage failed after {storage_time:.2f}s: {str(e)}")
            # Try to flush any buffered chunks before re-raising
            try:
                self.vector_store.flush_all_buffers()
            except Exception:
                pass
            raise StorageError(f"Failed to store document chunks: {str(e)}")

    def _compile_upload_result(
        self,
        document_id: str,
        filename: str,
        validation_result: Dict[str, Any],
        processing_result: Dict[str, Any],
        storage_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile final upload result."""
        total_processing_time = (
            processing_result['processing_time'] +
            storage_result.get('storage_time', 0)
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "total_chunks": len(processing_result['chunks']),
            "total_pages": validation_result['page_count'],
            "processing_time_seconds": total_processing_time,
        }
