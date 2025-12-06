"""Structured logging configuration."""
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict

from pydantic_settings import BaseSettings


class LogSettings(BaseSettings):
    """Logging configuration settings."""

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


def setup_logger() -> logging.Logger:
    """Configure structured JSON logging."""
    settings = LogSettings()

    class JSONFormatter(logging.Formatter):
        """Custom JSON formatter for structured logging."""

        def format(self, record: logging.LogRecord) -> str:
            log_data: Dict[str, Any] = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }

            # Add extra fields if present
            if hasattr(record, "retrieval_score"):
                log_data["retrieval_score"] = record.retrieval_score
            if hasattr(record, "llm_response_time"):
                log_data["llm_response_time"] = record.llm_response_time
            if hasattr(record, "token_usage"):
                log_data["token_usage"] = record.token_usage
            if hasattr(record, "answer_length"):
                log_data["answer_length"] = record.answer_length
            if hasattr(record, "document_id"):
                log_data["document_id"] = record.document_id
            if hasattr(record, "similarity_scores"):
                log_data["similarity_scores"] = record.similarity_scores

            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)

            return json.dumps(log_data)

    logger = logging.getLogger("smart_document_qa")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger


# Global logger instance
logger = setup_logger()

