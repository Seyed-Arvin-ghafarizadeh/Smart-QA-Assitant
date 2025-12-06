"""Script to pre-download the Sentence Transformer model locally."""
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = Path(__file__).parent / "models" / MODEL_NAME.replace("/", "_")


def download_model():
    """Download the model to a local directory."""
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Target directory: {MODEL_DIR}")
    
    # Create directory if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download the model
    print("Loading model (this may take a few minutes)...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Save the model locally
    print(f"Saving model to {MODEL_DIR}...")
    model.save(str(MODEL_DIR))
    
    print(f"Model successfully downloaded and saved to {MODEL_DIR}")
    print(f"You can now use this local path in your application: {MODEL_DIR}")


if __name__ == "__main__":
    download_model()
