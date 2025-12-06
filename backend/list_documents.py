"""Utility script to list all uploaded documents in ChromaDB."""
import chromadb
from chromadb.config import Settings
import os
import sys

def list_documents(db_path: str = "./chroma_db"):
    """
    List all documents stored in ChromaDB.
    
    Args:
        db_path: Path to ChromaDB directory
    """
    if not os.path.exists(db_path):
        print(f"❌ ChromaDB directory not found: {db_path}")
        print(f"   Make sure you're running this from the backend/ directory")
        return
    
    try:
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        collections = client.list_collections()
        
        if not collections:
            print("📭 No documents found in the database.")
            print("\n💡 Upload a document using:")
            print("   POST http://localhost:8000/api/upload")
            return
        
        print(f"\n📚 Found {len(collections)} document(s):\n")
        print("=" * 80)
        
        for idx, collection in enumerate(collections, 1):
            # Collection names are in format "doc_<uuid>"
            doc_id = collection.name.replace("doc_", "")
            chunk_count = collection.count()
            metadata = collection.metadata or {}
            
            print(f"\n{idx}. Document ID: {doc_id}")
            print(f"   Collection: {collection.name}")
            print(f"   Total Chunks: {chunk_count}")
            
            if metadata:
                print(f"   Metadata: {metadata}")
            
            # Get a sample chunk to show document preview
            try:
                sample = collection.get(limit=1, include=["documents", "metadatas"])
                if sample["documents"]:
                    preview = sample["documents"][0][:150]
                    print(f"   Preview: {preview}...")
                    if sample["metadatas"]:
                        page = sample["metadatas"][0].get("page_number", "N/A")
                        print(f"   First Chunk Page: {page}")
            except Exception as e:
                print(f"   (Could not load preview: {str(e)})")
            
            print("-" * 80)
        
        print(f"\n✅ Total: {len(collections)} document(s)")
        print("\n💡 To query a document, use:")
        print("   POST http://localhost:8000/api/ask")
        print('   Body: {"document_id": "<ID>", "question": "Your question here"}')
        
    except Exception as e:
        print(f"❌ Error accessing ChromaDB: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="List all documents in ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python list_documents.py
  python list_documents.py --db-path ./chroma_db
  python list_documents.py --db-path D:/Portfolio/Smart Document QA Assistant/backend/chroma_db
        """
    )
    parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="Path to ChromaDB directory (default: ./chroma_db)"
    )
    
    args = parser.parse_args()
    
    print("🔍 Scanning ChromaDB for documents...")
    list_documents(args.db_path)

