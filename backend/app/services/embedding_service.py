"""Embedding service using Sentence Transformers."""
import os
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

from app.utils.logger import logger


class EmbeddingService:
    """Service for generating text embeddings using Sentence Transformers."""

    _instance = None
    _model = None

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Singleton pattern to load model only once."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._model_name = model_name
        return cls._instance

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding service and load model."""
        if self._model is None:
            # Check for local model path (pre-downloaded)
            local_model_path = os.getenv("EMBEDDING_MODEL_PATH")
            
            # Try multiple local paths if EMBEDDING_MODEL_PATH not set
            if not local_model_path:
                possible_paths = [
                    # Docker container path
                    "/app/models/all-MiniLM-L6-v2",
                    # Local development paths (relative to backend/)
                    os.path.join(os.path.dirname(__file__), "..", "..", "models", "sentence-transformers_all-MiniLM-L6-v2"),
                    # Absolute path relative to current working directory
                    os.path.join(os.getcwd(), "models", "sentence-transformers_all-MiniLM-L6-v2"),
                    # Parent directory if running from backend/
                    os.path.join(os.path.dirname(os.getcwd()), "backend", "models", "sentence-transformers_all-MiniLM-L6-v2"),
                ]
                
                # Find first existing path
                for path in possible_paths:
                    abs_path = os.path.abspath(path)
                    if os.path.exists(abs_path) and os.path.isdir(abs_path):
                        local_model_path = abs_path
                        logger.info(f"Found local model at: {local_model_path}")
                        break
            
            # Use local model if it exists, otherwise use model_name (will download from HuggingFace)
            if local_model_path and os.path.exists(local_model_path) and os.path.isdir(local_model_path):
                logger.info(f"Loading embedding model from local path: {local_model_path}")
                self._model = SentenceTransformer(local_model_path)
                logger.info("Embedding model loaded successfully from local path")
            else:
                logger.info(f"Local model not found, loading from HuggingFace: {model_name}")
                logger.info("Note: This will download the model (~80MB). To use a local model, set EMBEDDING_MODEL_PATH environment variable.")
                self._model = SentenceTransformer(model_name)
                logger.info("Embedding model loaded successfully from HuggingFace")

    @property
    def model(self) -> SentenceTransformer:
        """Get the loaded model."""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        return self._model

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []

        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            # Convert numpy array to list of lists
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector as a list of floats
        """
        return self.generate_embeddings([text])[0]

