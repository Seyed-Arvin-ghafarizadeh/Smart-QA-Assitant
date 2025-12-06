"""Vector store service using ChromaDB."""
import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from app.models.document import Chunk
from app.utils.logger import logger


class VectorStore:
    """Service for storing and retrieving document embeddings in ChromaDB."""

    def __init__(self, db_path: str = "./chroma_db"):
        """
        Initialize ChromaDB client.

        Args:
            db_path: Path to ChromaDB persistent storage directory
        """
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

        logger.info(f"ChromaDB initialized at {db_path}")

    def _get_collection_name(self, document_id: str) -> str:
        """Generate collection name for a document."""
        return f"doc_{document_id}"

    def store_document(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Store document chunks and embeddings in ChromaDB.

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

        # Get or create collection
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"document_id": document_id, "num_chunks": len(chunks)},
        )

        # Prepare data for ChromaDB
        ids = [f"{document_id}_chunk_{chunk.chunk_index}" for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "document_id": document_id,
            }
            for chunk in chunks
        ]

        # Add to collection
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        logger.info(f"Stored {len(chunks)} chunks for document {document_id}")

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
            collection = self.client.get_collection(name=collection_name)
        except Exception:
            logger.warning(f"Collection not found for document {document_id}")
            return []

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
        )

        # Format results
        retrieved_chunks = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                # Convert distance to similarity and clamp to [0, 1]
                distance = results["distances"][0][i]
                similarity = max(0.0, min(1.0, 1.0 - distance))
                
                chunk_data = {
                    "text": results["documents"][0][i],
                    "page_number": results["metadatas"][0][i]["page_number"],
                    "chunk_index": results["metadatas"][0][i]["chunk_index"],
                    "similarity_score": similarity,
                    "document_id": document_id,
                }
                retrieved_chunks.append(chunk_data)

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for document {document_id}")
        return retrieved_chunks

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
        collections = self.client.list_collections()

        if not collections:
            logger.warning("No documents found in the database")
            return []

        # Search across all collections
        for collection in collections:
            try:
                # Query each collection
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=collection.count(),  # Get all chunks from each collection
                )

                # Format results
                if results["ids"] and len(results["ids"][0]) > 0:
                    document_id = collection.metadata.get("document_id", "unknown") if collection.metadata else "unknown"
                    for i in range(len(results["ids"][0])):
                        # Convert distance to similarity and clamp to [0, 1]
                        distance = results["distances"][0][i]
                        similarity = max(0.0, min(1.0, 1.0 - distance))
                        
                        chunk_data = {
                            "text": results["documents"][0][i],
                            "page_number": results["metadatas"][0][i]["page_number"],
                            "chunk_index": results["metadatas"][0][i]["chunk_index"],
                            "similarity_score": similarity,
                            "document_id": document_id,
                        }
                        all_chunks.append(chunk_data)
            except Exception as e:
                logger.warning(f"Error querying collection {collection.name}: {str(e)}")
                continue

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
            collection = self.client.get_collection(name=collection_name)
            return collection.count() > 0
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
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted document {document_id} from vector store")
        except Exception as e:
            logger.warning(f"Error deleting document {document_id}: {str(e)}")

