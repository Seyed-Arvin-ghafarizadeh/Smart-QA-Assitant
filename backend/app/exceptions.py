"""Custom exception classes for document processing."""


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass


class ValidationError(DocumentProcessingError):
    """Raised when document validation fails."""
    pass


class FileTypeNotSupportedError(ValidationError):
    """Raised when an unsupported file type is encountered."""
    pass


class FileSizeExceededError(ValidationError):
    """Raised when file size exceeds the maximum allowed."""
    pass


class DocumentCorruptedError(ValidationError):
    """Raised when a document file appears to be corrupted."""
    pass


class DocumentEmptyError(ValidationError):
    """Raised when a document has no extractable content."""
    pass


class PageLimitExceededError(ValidationError):
    """Raised when document exceeds maximum page limit."""
    pass


class ChunkLimitExceededError(ValidationError):
    """Raised when document would create too many chunks."""
    pass


class ProcessingError(DocumentProcessingError):
    """Raised when document processing fails."""
    pass


class ExtractionError(ProcessingError):
    """Raised when text extraction from document fails."""
    pass


class EmbeddingError(DocumentProcessingError):
    """Raised when embedding generation fails."""
    pass


class StorageError(DocumentProcessingError):
    """Raised when storing document chunks fails."""
    pass


class ServiceUnavailableError(DocumentProcessingError):
    """Raised when required services are not available."""
    pass
