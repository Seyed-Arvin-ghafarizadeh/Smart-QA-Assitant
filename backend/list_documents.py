"""Utility script to list all uploaded documents in Qdrant."""
from qdrant_client import QdrantClient
import os
import sys

def list_documents(db_path: str = "./qdrant_db"):
    """
    List all documents stored in Qdrant.
    
    Args:
        db_path: Path to Qdrant directory
    """
    if not os.path.exists(db_path):
        print(f"❌ Qdrant directory not found: {db_path}")
        print(f"   Make sure you're running this from the backend/ directory")
        return
    
    try:
        client = QdrantClient(path=db_path)
        
        collections = client.get_collections().collections
        
        if not collections:
            print("📭 No documents found in the database.")
            print("\n💡 Upload a document using:")
            print("   POST http://localhost:8000/api/upload")
            return
        
        # Expected dimension for all-MiniLM-L6-v2
        EXPECTED_DIM = 384
        
        print(f"\n📚 Found {len(collections)} document(s):\n")
        print(f"ℹ️  Expected vector dimension: {EXPECTED_DIM} (all-MiniLM-L6-v2)")
        print("=" * 80)
        
        dimension_issues = []
        
        for idx, collection in enumerate(collections, 1):
            # Collection names are in format "doc_<uuid>"
            doc_id = collection.name.replace("doc_", "")
            
            # Get collection info
            try:
                collection_info = client.get_collection(collection.name)
                chunk_count = collection_info.points_count
                
                # Get vector dimension
                vectors_config = collection_info.config.params.vectors
                if hasattr(vectors_config, 'size'):
                    vector_dim = vectors_config.size
                elif isinstance(vectors_config, dict):
                    first_vector = next(iter(vectors_config.values()))
                    vector_dim = first_vector.size if hasattr(first_vector, 'size') else "unknown"
                else:
                    vector_dim = "unknown"
            except Exception as e:
                chunk_count = 0
                vector_dim = f"error: {e}"
            
            # Check for dimension mismatch
            dim_status = "✅" if vector_dim == EXPECTED_DIM else "⚠️ MISMATCH"
            if vector_dim != EXPECTED_DIM and vector_dim != "unknown":
                dimension_issues.append((doc_id, vector_dim))
            
            print(f"\n{idx}. Document ID: {doc_id}")
            print(f"   Collection: {collection.name}")
            print(f"   Total Chunks: {chunk_count}")
            print(f"   Vector Dimension: {vector_dim} {dim_status}")
            
            # Get a sample chunk to show document preview
            try:
                results = client.scroll(
                    collection_name=collection.name,
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )
                if results[0]:  # results is (points, next_page_offset)
                    point = results[0][0]
                    payload = point.payload
                    text = payload.get("text", "")
                    preview = text[:150] if text else "No text available"
                    print(f"   Preview: {preview}...")
                    page = payload.get("page_number", "N/A")
                    print(f"   First Chunk Page: {page}")
            except Exception as e:
                print(f"   (Could not load preview: {str(e)})")
            
            print("-" * 80)
        
        # Show dimension mismatch warning
        if dimension_issues:
            print("\n" + "=" * 80)
            print("⚠️  DIMENSION MISMATCH DETECTED!")
            print("=" * 80)
            print(f"Expected dimension: {EXPECTED_DIM} (all-MiniLM-L6-v2)")
            print("Documents with mismatched dimensions:")
            for doc_id, dim in dimension_issues:
                print(f"  - {doc_id}: dimension {dim}")
            print("\n🔧 To fix: Delete these documents and re-upload them.")
            print("   You can delete a document using the /api/documents/{id} DELETE endpoint.")
        
        print(f"\n✅ Total: {len(collections)} document(s)")
        print("\n💡 To query a document, use:")
        print("   POST http://localhost:8000/api/ask")
        print('   Body: {"question": "Your question here"}')
        
    except Exception as e:
        print(f"❌ Error accessing Qdrant: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="List all documents in Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python list_documents.py
  python list_documents.py --db-path ./qdrant_db
  python list_documents.py --db-path D:/Portfolio/Smart Document QA Assistant/backend/qdrant_db
        """
    )
    parser.add_argument(
        "--db-path",
        default="./qdrant_db",
        help="Path to Qdrant directory (default: ./qdrant_db)"
    )
    
    args = parser.parse_args()
    
    print("🔍 Scanning Qdrant for documents...")
    list_documents(args.db_path)
