"""Embedding service using Sentence Transformers."""
import os
import threading
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer
import torch

from app.utils.logger import logger


def configure_cpu_cores(cpu_cores: int = 0) -> int:
    """
    Configure the number of CPU cores for PyTorch.
    
    Args:
        cpu_cores: Number of CPU cores to use (0 = use all available)
        
    Returns:
        Actual number of cores configured
    """
    available_cores = os.cpu_count() or 1
    cores_to_use = cpu_cores if cpu_cores > 0 else available_cores
    
    # Configure PyTorch threading for optimal CPU performance
    # Use all available cores for better parallelization
    torch.set_num_threads(cores_to_use)
    # Increase interop threads for better parallel batch processing
    torch.set_num_interop_threads(max(1, cores_to_use))
    
    # Set environment variables for optimal CPU performance
    os.environ['OMP_NUM_THREADS'] = str(cores_to_use)
    os.environ['MKL_NUM_THREADS'] = str(cores_to_use)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cores_to_use)
    
    logger.info(f"CPU configuration: using {cores_to_use} cores (available: {available_cores})")
    return cores_to_use


# Global model cache for worker processes (each process has its own)
# Note: Each process will have its own copy of this variable
_worker_model_cache = None
_worker_model_lock = threading.Lock()  # Thread lock is fine since each process has its own


def _get_model_path(model_name: str) -> Optional[str]:
    """Get local model path if available."""
    local_model_path = os.getenv("EMBEDDING_MODEL_PATH")
    
    if not local_model_path:
        possible_paths = [
            "/app/models/all-MiniLM-L6-v2",
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "sentence-transformers_all-MiniLM-L6-v2"),
            os.path.join(os.getcwd(), "models", "sentence-transformers_all-MiniLM-L6-v2"),
            os.path.join(os.path.dirname(os.getcwd()), "backend", "models", "sentence-transformers_all-MiniLM-L6-v2"),
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) and os.path.isdir(abs_path):
                return abs_path
    
    return local_model_path if local_model_path and os.path.exists(local_model_path) else None


def _load_worker_model(model_name: str, model_path: Optional[str] = None, max_workers: int = 0) -> SentenceTransformer:
    """
    Load model in a worker process (called once per process).
    This avoids pickling issues with ProcessPoolExecutor.
    Each process will load its own copy of the model.
    
    Args:
        model_name: Name or path to the model
        model_path: Optional local path to model
        max_workers: Number of parallel workers (for thread optimization)
    """
    global _worker_model_cache
    
    if _worker_model_cache is None:
        with _worker_model_lock:
            if _worker_model_cache is None:
                # Configure CPU for this process
                # When using multiple processes, limit threads per process to avoid oversubscription
                available_cores = os.cpu_count() or 1
                if max_workers > 1:
                    # Use fewer threads per process (total threads = processes * threads_per_process)
                    threads_per_process = max(1, available_cores // max_workers)
                    torch.set_num_threads(threads_per_process)
                    torch.set_num_interop_threads(1)
                else:
                    # Use all cores for single process
                    configure_cpu_cores(0)
                
                device = "cpu"
                if model_path and os.path.exists(model_path):
                    _worker_model_cache = SentenceTransformer(model_path, device=device)
                else:
                    _worker_model_cache = SentenceTransformer(model_name, device=device)
    
    return _worker_model_cache


def _generate_embeddings_worker(args: tuple) -> List[List[float]]:
    """
    Worker function for parallel embedding generation.
    Loads model in each process to avoid pickling issues.
    
    Args:
        args: Tuple of (model_name, model_path, texts, batch_size, max_workers)
        
    Returns:
        List of embedding vectors
    """
    model_name, model_path, texts, batch_size, max_workers = args
    
    if not texts:
        return []
    
    try:
        model = _load_worker_model(model_name, model_path, max_workers)
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            device="cpu",
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        
        return embeddings.tolist()
    except Exception as e:
        # Log error but return empty list to avoid crashing the process pool
        import sys
        print(f"Error in worker process: {str(e)}", file=sys.stderr)
        raise


class EmbeddingService:
    """Service for generating text embeddings using Sentence Transformers."""

    _instance = None
    _model = None
    _lock = threading.Lock()
    _cpu_cores = 0
    _device = None
    _batch_size = 128
    _use_parallel = True
    _max_parallel_workers = 0  # 0 = auto (CPU cores)

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cpu_cores: int = 0, batch_size: int = 128, use_parallel: bool = True, max_parallel_workers: int = 0):
        """Singleton pattern to load model only once."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._model_name = model_name
            cls._cpu_cores = cpu_cores
            cls._batch_size = batch_size
            cls._use_parallel = use_parallel
            cls._max_parallel_workers = max_parallel_workers
        return cls._instance

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cpu_cores: int = 0, batch_size: int = 128, use_parallel: bool = True, max_parallel_workers: int = 0):
        """
        Initialize embedding service (model loaded lazily on first use).
        
        Args:
            model_name: Name of the sentence transformer model
            cpu_cores: Number of CPU cores to use (0 = use all available)
            batch_size: Batch size for embedding generation
            use_parallel: Whether to use parallel processing for multiple batches
            max_parallel_workers: Max workers for parallel processing (0 = auto: CPU cores)
        """
        # Store model name but don't load model yet
        if not hasattr(self, '_model_name'):
            self._model_name = model_name
            self._cpu_cores = cpu_cores
            self._batch_size = batch_size
            self._use_parallel = use_parallel
            self._max_parallel_workers = max_parallel_workers
            logger.info(f"EmbeddingService initialized (model will be loaded on first use, parallel={use_parallel})")

    def _load_model(self) -> SentenceTransformer:
        """Load the model (thread-safe lazy loading) with optimized CPU/GPU configuration."""
        # Double-check locking pattern for thread safety
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("Loading embedding model (lazy initialization)...")
                    
                    # Configure CPU cores for optimal performance
                    configure_cpu_cores(self._cpu_cores)
                    
                    # Force CPU-only mode (no GPU checks to avoid delays)
                    self._device = "cpu"
                    logger.info(f"Using CPU for embeddings (CPU-only mode)")
                    
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
                        self._model = SentenceTransformer(local_model_path, device=self._device)
                        logger.info("Embedding model loaded successfully from local path")
                    else:
                        logger.info(f"Local model not found, loading from HuggingFace: {self._model_name}")
                        logger.info("Note: This will download the model (~80MB). To use a local model, set EMBEDDING_MODEL_PATH environment variable.")
                        self._model = SentenceTransformer(self._model_name, device=self._device)
                        logger.info("Embedding model loaded successfully from HuggingFace")
                    
                    logger.info(f"Embedding model ready on device: {self._device}, batch_size: {self._batch_size}")
        return self._model

    @property
    def model(self) -> SentenceTransformer:
        """Get the loaded model (loads lazily on first access)."""
        return self._load_model()

    def generate_embeddings(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with optimized batch processing.

        Args:
            texts: List of text strings to embed
            batch_size: Optional batch size (uses configured default if not provided)

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []

        try:
            # Use provided batch_size or fall back to configured default
            effective_batch_size = batch_size or self._batch_size
            
            # Optimize encoding for CPU with better parallelization
            embeddings = self.model.encode(
                texts, 
                batch_size=effective_batch_size,
                show_progress_bar=False,
                device=self._device,
                convert_to_numpy=True,
                normalize_embeddings=False,  # Slightly faster
            )
            # Convert numpy array to list of lists
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise

    def generate_embeddings_parallel_batches(self, text_batches: List[List[str]], batch_size: Optional[int] = None) -> List[List[List[float]]]:
        """
        Generate embeddings for multiple batches in parallel using ProcessPoolExecutor.
        This bypasses the GIL and provides true CPU parallelism.
        
        Args:
            text_batches: List of text batches (each batch is a list of strings)
            batch_size: Optional batch size for each batch (uses configured default if not provided)
            
        Returns:
            List of embedding batches (each batch is a list of embedding vectors)
        """
        if not text_batches:
            return []
        
        # Filter out empty batches
        text_batches = [batch for batch in text_batches if batch]
        if not text_batches:
            return []
        
        # For single batch or if parallel is disabled, use sequential processing
        if len(text_batches) == 1 or not self._use_parallel:
            return [self.generate_embeddings(batch, batch_size) for batch in text_batches]
        
        # Determine number of workers
        available_cores = os.cpu_count() or 1
        max_workers = self._max_parallel_workers if self._max_parallel_workers > 0 else available_cores
        # Limit workers to avoid overhead (don't use more workers than batches)
        max_workers = min(max_workers, len(text_batches), available_cores)
        
        # For small number of batches, sequential might be faster (avoid process overhead)
        if len(text_batches) <= 2:
            return [self.generate_embeddings(batch, batch_size) for batch in text_batches]
        
        logger.info(f"⚡ Processing {len(text_batches)} embedding batches in PARALLEL using {max_workers} workers")
        
        try:
            # Get model path for workers
            model_path = _get_model_path(self._model_name)
            effective_batch_size = batch_size or self._batch_size
            
            # Prepare arguments for workers
            worker_args = [
                (self._model_name, model_path, batch, effective_batch_size, max_workers)
                for batch in text_batches
            ]
            
            # Process batches in parallel
            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {executor.submit(_generate_embeddings_worker, args): idx 
                          for idx, args in enumerate(worker_args)}
                
                # Collect results as they complete (maintain order)
                results_dict = {}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results_dict[idx] = result
                    except Exception as e:
                        logger.error(f"Error in parallel embedding worker {idx}: {str(e)}", exc_info=True)
                        # Fallback to sequential for this batch
                        results_dict[idx] = self.generate_embeddings(worker_args[idx][2], effective_batch_size)
                
                # Reconstruct results in original order
                results = [results_dict[i] for i in range(len(worker_args))]
            
            logger.info(f"✅ Completed parallel embedding generation for {len(text_batches)} batches")
            return results
            
        except Exception as e:
            logger.warning(f"Parallel embedding generation failed, falling back to sequential: {str(e)}")
            # Fallback to sequential processing
            return [self.generate_embeddings(batch, batch_size) for batch in text_batches]

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector as a list of floats
        """
        return self.generate_embeddings([text])[0]

