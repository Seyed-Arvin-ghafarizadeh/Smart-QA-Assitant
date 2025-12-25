"""Vector store service using Qdrant."""
import os
import hashlib
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

from app.models.document import Chunk
from app.utils.logger import logger


class VectorStore:
    """Service for storing and retrieving document embeddings in Qdrant."""

    def __init__(self, db_path: str = "./qdrant_db", batch_size: int = 100):
        """
        Initialize Qdrant client.

        Args:
            db_path: Path to Qdrant persistent storage directory
            batch_size: Maximum batch size for storing chunks (default: 100)
        """
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)

        # Use Qdrant in local/embedded mode for persistent storage
        self.client = QdrantClient(path=db_path)

        self.batch_size = batch_size
        # Internal buffer for single-chunk storage batching
        self._chunk_buffer: dict = {}  # document_id -> list of (chunk, embedding) tuples
        logger.info(f"Qdrant initialized at {db_path} with batch size {batch_size}")

    def _get_collection_name(self, document_id: str) -> str:
        """Generate collection name for a document."""
        return f"doc_{document_id}"
    
    def _generate_point_id(self, document_id: str, chunk_index: int) -> int:
        """
        Generate a unique integer ID for a point.
        
        Args:
            document_id: Document ID
            chunk_index: Chunk index
            
        Returns:
            Unique integer ID
        """
        # Create a hash from document_id + chunk_index to get a unique integer
        # Use first 8 bytes of hash as uint64
        combined = f"{document_id}_{chunk_index}".encode('utf-8')
        hash_obj = hashlib.md5(combined)
        # Take first 8 bytes and convert to int (uint64)
        hash_bytes = hash_obj.digest()[:8]
        point_id = int.from_bytes(hash_bytes, byteorder='big')
        # Ensure it's positive (uint64)
        return point_id & 0x7FFFFFFFFFFFFFFF

    def _ensure_collection(self, collection_name: str, vector_size: int = 384):
        """
        Ensure collection exists, create if it doesn't.

        Args:
            collection_name: Name of the collection
            vector_size: Size of embedding vectors (default: 384 for all-MiniLM-L6-v2)
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                # Create collection with cosine similarity
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.debug(f"Created Qdrant collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name}: {str(e)}", exc_info=True)
            raise

    def store_chunk(self, chunk: Chunk, embedding: List[float]) -> None:
        """
        Store a single chunk and embedding in Qdrant (optimized for streaming).

        Uses internal buffering to batch chunks for efficient storage.
        Automatically flushes when buffer reaches batch_size.

        Args:
            chunk: Chunk object with metadata
            embedding: Embedding vector for the chunk
        """
        document_id = chunk.document_id
        
        # Initialize buffer for this document if needed
        if document_id not in self._chunk_buffer:
            self._chunk_buffer[document_id] = []
        
        # Add chunk to buffer
        self._chunk_buffer[document_id].append((chunk, embedding))
        
        # Flush buffer if it reaches batch size
        if len(self._chunk_buffer[document_id]) >= self.batch_size:
            self._flush_buffer(document_id)

    def _flush_buffer(self, document_id: str) -> None:
        """
        Flush buffered chunks for a document to Qdrant.

        Args:
            document_id: Document ID to flush
        """
        if document_id not in self._chunk_buffer or not self._chunk_buffer[document_id]:
            return

        chunks_and_embeddings = self._chunk_buffer[document_id]
        chunks = [c[0] for c in chunks_and_embeddings]
        embeddings = [c[1] for c in chunks_and_embeddings]
        
        # Clear buffer
        self._chunk_buffer[document_id] = []
        
        # Store using batch method
        self.store_document(chunks, embeddings)

    def flush_all_buffers(self) -> None:
        """
        Flush all buffered chunks to Qdrant.
        Useful for ensuring all chunks are stored before finishing.
        """
        for document_id in list(self._chunk_buffer.keys()):
            self._flush_buffer(document_id)

    def store_document(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Store document chunks and embeddings in Qdrant using batch processing.

        Args:
            chunks: List of Chunk objects with metadata
            embeddings: List of embedding vectors
        """
        if not chunks or not embeddings:
            raise ValueError("Chunks and embeddings cannot be empty")
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        document_id = chunks[0].document_id
        collection_name = self._get_collection_name(document_id)
        
        # Get vector size from first embedding
        vector_size = len(embeddings[0])
        
        # Ensure collection exists
        self._ensure_collection(collection_name, vector_size)

        total_chunks = len(chunks)

        # Process in batches if needed
        if total_chunks <= self.batch_size:
            # Small document, process all at once
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                payload = {
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "document_id": document_id,
                }
                # Only add chapter_number if it's not None
                if chunk.chapter_number is not None:
                    payload["chapter_number"] = chunk.chapter_number
                
                point_id = self._generate_point_id(document_id, chunk.chunk_index)
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                )

            self.client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Stored {total_chunks} chunks for document {document_id} in 1 batch")
        else:
            # Large document, process in batches
            num_batches = (total_chunks + self.batch_size - 1) // self.batch_size
            logger.info(
                f"Processing {total_chunks} chunks for document {document_id} in {num_batches} batches "
                f"(batch size: {self.batch_size})"
            )

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_chunks)

                batch_chunks = chunks[start_idx:end_idx]
                batch_embeddings = embeddings[start_idx:end_idx]

                # Prepare batch points
                points = []
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    payload = {
                        "text": chunk.text,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "document_id": document_id,
                    }
                    # Only add chapter_number if it's not None
                    if chunk.chapter_number is not None:
                        payload["chapter_number"] = chunk.chapter_number
                    
                    point_id = self._generate_point_id(document_id, chunk.chunk_index)
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload
                        )
                    )

                # Upsert batch to collection
                try:
                    self.client.upsert(collection_name=collection_name, points=points)
                    logger.info(
                        f"Stored batch {batch_idx + 1}/{num_batches} "
                        f"({end_idx - start_idx} chunks) for document {document_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error storing batch {batch_idx + 1}/{num_batches} for document {document_id}: {str(e)}",
                        exc_info=True,
                    )
                    raise

            logger.info(f"Successfully stored all {total_chunks} chunks for document {document_id}")

    def retrieve_similar(
        self, query_embedding: List[float], document_id: str, top_k: int = 5
    ) -> List[dict]:
        """
        Retrieve top-k similar chunks for a query.

        Args:
            query_embedding: Query embedding vector
            document_id: Document ID to search within
            top_k: Number of chunks to retrieve

        Returns:
            List of dictionaries with chunk data and similarity scores
        """
        collection_name = self._get_collection_name(document_id)

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                logger.warning(f"Collection not found for document {document_id}")
                return []

            # Query Qdrant using query_points (new API in qdrant-client 1.7+)
            query_result = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=top_k,
            )

            # Format results
            retrieved_chunks = []
            for point in query_result.points:
                # Qdrant returns score (higher is better, cosine similarity)
                # Score is already in [0, 1] range for cosine distance
                similarity = float(point.score)
                
                payload = point.payload
                chunk_data = {
                    "text": payload.get("text", ""),
                    "page_number": payload.get("page_number", 0),
                    "chunk_index": payload.get("chunk_index", 0),
                    "similarity_score": similarity,
                    "document_id": document_id,
                    "chapter_number": payload.get("chapter_number"),
                }
                retrieved_chunks.append(chunk_data)

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for document {document_id}")
            return retrieved_chunks

        except Exception as e:
            logger.warning(f"Error querying collection {collection_name}: {str(e)}")
            return []

    def retrieve_similar_across_all(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[dict]:
        """
        Retrieve top-k similar chunks across all documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of chunks to retrieve

        Returns:
            List of dictionaries with chunk data and similarity scores
        """
        all_chunks = []
        query_dim = len(query_embedding)
        
        try:
            collections = self.client.get_collections().collections
        except Exception as e:
            logger.warning(f"Error listing collections: {str(e)}")
            return []

        if not collections:
            logger.warning("No documents found in the database")
            return []

        # Track collection stats for diagnostics
        total_collections = len(collections)
        collections_with_points = 0
        total_points = 0

        # Search across all collections
        dimension_mismatches = []
        for collection in collections:
            try:
                # Check if collection has any points before searching
                collection_info = self.client.get_collection(collection.name)
                points_count = collection_info.points_count
                total_points += points_count
                
                if points_count == 0:
                    logger.debug(f"Collection {collection.name} exists but has no points, skipping")
                    continue
                
                collections_with_points += 1
                
                # Check vector dimension mismatch - handle different Qdrant API versions
                try:
                    vectors_config = collection_info.config.params.vectors
                    # Handle both VectorParams (single vector) and dict (named vectors)
                    if hasattr(vectors_config, 'size'):
                        vector_size = vectors_config.size
                    elif isinstance(vectors_config, dict):
                        # Named vectors - get the first one
                        first_vector = next(iter(vectors_config.values()))
                        vector_size = first_vector.size if hasattr(first_vector, 'size') else None
                    else:
                        vector_size = None
                        
                    if vector_size and vector_size != query_dim:
                        dimension_mismatches.append(
                            f"{collection.name}: expected {vector_size}, got {query_dim}"
                        )
                        logger.error(
                            f"Vector dimension mismatch: query embedding has {query_dim} dimensions, "
                            f"but collection {collection.name} expects {vector_size} dimensions. "
                            f"This usually means the embedding model was changed. "
                            f"Please delete old documents and re-upload them."
                        )
                        continue
                except Exception as dim_err:
                    logger.warning(f"Could not check vector dimensions for {collection.name}: {dim_err}")
                    # Continue anyway and let Qdrant handle it
                
                # Query each collection using query_points (new API in qdrant-client 1.7+)
                # We'll sort and take top_k across all documents later
                logger.debug(f"Searching collection {collection.name} with {points_count} points")
                query_result = self.client.query_points(
                    collection_name=collection.name,
                    query=query_embedding,
                    limit=top_k * 2,  # Get more results per collection
                )
                
                logger.debug(f"Got {len(query_result.points)} results from collection {collection.name}")

                # Format results
                for point in query_result.points:
                    similarity = float(point.score)
                    payload = point.payload
                    
                    # Extract document_id from collection name or payload
                    document_id = payload.get("document_id")
                    if not document_id:
                        # Try to extract from collection name: doc_{document_id}
                        if collection.name.startswith("doc_"):
                            document_id = collection.name[4:]
                        else:
                            document_id = "unknown"
                    
                    chunk_data = {
                        "text": payload.get("text", ""),
                        "page_number": payload.get("page_number", 0),
                        "chunk_index": payload.get("chunk_index", 0),
                        "similarity_score": similarity,
                        "document_id": document_id,
                        "chapter_number": payload.get("chapter_number"),
                    }
                    all_chunks.append(chunk_data)
            except Exception as e:
                logger.warning(f"Error querying collection {collection.name}: {str(e)}", exc_info=True)
                continue

        # Log diagnostic information if no chunks found
        if not all_chunks:
            diag_msg = (
                f"Retrieved 0 chunks. Diagnostics: "
                f"{total_collections} collections found, "
                f"{collections_with_points} collections with points, "
                f"{total_points} total points across all collections, "
                f"query embedding dimension: {query_dim}"
            )
            if dimension_mismatches:
                diag_msg += f", DIMENSION MISMATCHES: {dimension_mismatches}"
            logger.warning(diag_msg)
            return []

        # Sort by similarity score (descending) and take top_k
        all_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_chunks = all_chunks[:top_k]
        logger.info(f"Retrieved {len(top_chunks)} chunks across all documents")
        return top_chunks

    def document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists in the vector store.

        Args:
            document_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """
        collection_name = self._get_collection_name(document_id)
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            if collection_name not in collection_names:
                return False
            
            # Check if collection has any points
            collection_info = self.client.get_collection(collection_name)
            return collection_info.points_count > 0
        except Exception:
            return False

    def delete_document(self, document_id: str) -> None:
        """
        Delete a document from the vector store.

        Args:
            document_id: Document ID to delete
        """
        collection_name = self._get_collection_name(document_id)
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted document {document_id} from vector store")
        except Exception as e:
            logger.warning(f"Error deleting document {document_id}: {str(e)}")
