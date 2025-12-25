# Smart Document QA Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that enables users to upload PDF, DOCX, and TXT documents and ask questions about them. Built with FastAPI, Sentence Transformers, Qdrant, Redis caching, and DeepSeek API.

## 🎯 Overview

This application implements a complete RAG pipeline with advanced features:

1. **Document Ingestion**: Extract text from PDFs, DOCX, and TXT files with OCR fallback support for scanned PDFs
2. **Text Processing**: Clean and chunk documents (500 chars with 100 char overlap) for optimal semantic search
3. **Embedding Generation**: Create vector embeddings using Sentence Transformers (all-MiniLM-L6-v2)
4. **Vector Storage**: Store embeddings and metadata in Qdrant (local/embedded mode)
5. **Query Expansion**: Automatically expand user queries using LLM to improve retrieval accuracy
6. **Retrieval**: Find relevant chunks using semantic similarity search with optimized thresholds
7. **Caching**: Redis-based caching for faster repeated queries
8. **Answer Generation**: Use DeepSeek API to generate grounded answers with citations

## 🏗️ Architecture

```
Client → FastAPI → Document Processor
                   ↓
             Embedding Generator → Qdrant Vector DB
                   ↓
         Query Expansion (LLM) → Embedding Generator
                   ↓
               Semantic Search → Redis Cache
                   ↓
               Answer Generator (LLM) → Response Builder → Client
```

### Components

- **Backend**: FastAPI service with RESTful API endpoints
- **ML Pipeline**: Document processing, embeddings, vector search, query expansion, LLM integration
- **Frontend**: Modern Streamlit web interface for document upload and Q&A
- **Vector DB**: Qdrant (local/embedded mode) for persistent storage of embeddings
- **Caching**: Redis for caching query results and improving response times
- **Query Expansion**: LLM-powered query expansion to improve retrieval accuracy
- **Monitoring**: Prometheus-compatible metrics, OpenTelemetry tracing, and structured logging

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional, for containerized deployment)
- DeepSeek API key
- **Poppler** (required for OCR functionality on scanned PDFs)
  - **Windows**: Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases) and add to PATH
  - **Linux**: `sudo apt-get install poppler-utils` (Ubuntu/Debian) or `sudo yum install poppler-utils` (RHEL/CentOS)
  - **macOS**: `brew install poppler`
  
  > **Note**: EasyOCR models (~200MB) will be automatically downloaded on first use.

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Smart Document QA Assistant"
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

5. **Configure environment variables**
   
   **IMPORTANT**: Create a `.env` file in the **root directory** (same level as `backend/` and `frontend/` folders).
   
   Copy the template below and replace `your_deepseek_api_key_here` with your actual DeepSeek API key:
   
   ```env
   # DeepSeek API Configuration (REQUIRED)
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
   
   # Qdrant Vector Database Configuration
   QDRANT_DB_PATH=./qdrant_db
   
   # Redis Cache Configuration (optional but recommended)
   REDIS_URL=redis://localhost:6379/0
   ENABLE_CACHE=true
   CACHE_TTL_SECONDS=3600
   
   # Backend API Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   
   # Frontend Configuration
   FRONTEND_PORT=80
   
   # Logging
   LOG_LEVEL=INFO
   
   # Retrieval Configuration
   TOP_K_CHUNKS=5
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=100
   
   # Embedding Model
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   
   # Document Upload Limits (for validation)
   MAX_FILE_SIZE_MB=50
   MAX_PAGES=1000
   MAX_CHARACTERS=2000000
   MAX_WORDS=300000
   MAX_CHUNKS=5000
   MAX_SENTENCES=10000
   
   # Processing Configuration
   QDRANT_BATCH_SIZE=500               # Qdrant batch size for storing chunks
   EMBEDDING_BATCH_SIZE=256            # Batch size for embedding generation
   
   # Similarity Threshold Configuration (for semantic search)
   SIMILARITY_MAX_THRESHOLD=0.20       # Maximum similarity threshold (lowered for better recall)
   SIMILARITY_AVG_THRESHOLD=0.15       # Average similarity threshold
   SIMILARITY_MIN_SCORE=0.15           # Minimum score for "high similarity" chunks
   
   # Query Expansion Configuration
   ENABLE_QUERY_EXPANSION=true         # Enable LLM-based query expansion to improve retrieval
   
   # Performance Optimization
   SKIP_RELEVANCE_CHECK=true           # Skip expensive LLM relevance check (saves 1 API call)
   SKIP_ANSWER_VALIDATION=true         # Skip expensive LLM answer validation (saves 1 API call)
   
   # OCR Configuration (EasyOCR for scanned PDFs)
   OCR_ENABLED=true                    # Enable/disable OCR fallback
   OCR_LANGUAGES=en                    # Comma-separated language codes (e.g., "en,fr,es" for English, French, Spanish)
   OCR_GPU=false                       # Use GPU for OCR (requires CUDA)
   OCR_DPI=300                         # DPI for OCR image conversion (higher = better quality but slower)
   OCR_TEXT_THRESHOLD=50               # Minimum characters before using OCR fallback
   ```
   
   > **Note**: See `SETUP_ENV.md` for detailed setup instructions.
   
   **Validation Limits Rationale** (optimized for mid-range hardware, 8-16GB RAM):
   - **MAX_FILE_SIZE_MB=50**: Prevents memory issues, reasonable for most documents
   - **MAX_PAGES=1000**: ~6-14 minutes processing time, manageable for typical use
   - **MAX_CHUNKS=5000**: Limits total chunks per document for vector database efficiency
   - **CHUNK_SIZE=500**: Optimal chunk size for semantic search (smaller chunks for better granularity)
   - **CHUNK_OVERLAP=100**: 20% overlap ensures context continuity between chunks
   - **QDRANT_BATCH_SIZE=500**: Optimized batch size for Qdrant storage
   - **EMBEDDING_BATCH_SIZE=256**: Optimized batch size for parallel embedding generation

6. **Run the backend**
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Serve the frontend**
   ```bash
   # Using Streamlit
   cd frontend
   streamlit run app.py
   
   # Or using Python HTTP server (if using static HTML)
   python -m http.server 8080
   ```

8. **Access the application**
   - Frontend: http://localhost:8080
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - Metrics: http://localhost:8000/metrics

### Docker Deployment

The Docker build process automatically pre-downloads the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) during the image build phase. This avoids proxy issues and ensures the model is available immediately when the container starts.

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```
   
   > **Note**: The first build may take several minutes as it downloads the model (~80MB). Subsequent builds will be faster due to Docker layer caching.

2. **Access the application**
   - Frontend: http://localhost
   - Backend API: http://localhost:8000

#### Manual Model Download (Optional)

If you want to pre-download the model locally before building the Docker image (useful for offline builds or to avoid proxy issues during build):

```bash
cd backend
python download_model.py
```

The model will be saved to `backend/models/sentence-transformers_all-MiniLM-L6-v2/`. You can then set the `EMBEDDING_MODEL_PATH` environment variable in `docker-compose.yml` to use this local path.

## 📡 API Endpoints

### POST `/api/upload`

Upload and process a document (PDF, DOCX, or TXT).

**Request:**
- `file`: Document file (multipart/form-data) - supports PDF, DOCX, and TXT formats

**Response:**
```json
{
  "document_id": "uuid-string",
  "filename": "document.pdf",
  "total_chunks": 42,
  "total_pages": 10,
  "message": "Document uploaded and processed successfully"
}
```

**Validation:**
The upload endpoint performs validation during streaming processing:
- File type check (must be PDF, DOCX, or TXT)
- File size validation (default max: 50MB)
- Document integrity check (verifies file is readable)
- Page/section count validation (stops early if exceeds max_pages)
- Chunk count validation (stops early if exceeds max_chunks during processing)
- OCR fallback for scanned PDFs (if text extraction yields insufficient text)

If any limit is exceeded, the upload is rejected with a clear error message indicating which limit was exceeded and at what point processing stopped.

### POST `/api/ask`

Ask a question about uploaded documents. Searches across all uploaded documents.

**Request:**
```json
{
  "question": "What is the main topic?"
}
```

**Response:**
```json
{
  "answer": "The main topic is... (3 paragraphs)",
  "relevant_chunks": [
    {
      "text": "Chunk text...",
      "page_number": 1,
      "chapter_number": 1,
      "similarity_score": 0.85,
      "document_id": "uuid-string"
    }
  ],
  "confidence": 0.82,
  "token_usage": {
    "prompt_tokens": 500,
    "completion_tokens": 150,
    "total_tokens": 650
  },
  "response_time_ms": 1250.5,
  "sentiment": "neutral",
  "is_relevant": true,
  "similarity_metrics": {
    "max_similarity": 0.85,
    "avg_similarity": 0.75,
    "min_similarity": 0.65,
    "high_similarity_count": 3
  }
}
```

**Features:**
- **Query Expansion**: Automatically expands queries using LLM to improve retrieval
- **Multi-Document Search**: Searches across all uploaded documents simultaneously
- **Caching**: Results are cached in Redis for faster repeated queries
- **Similarity Thresholds**: Optimized thresholds (0.20/0.15) for better recall

### GET `/api/metrics`

Get Prometheus-compatible metrics.

### GET `/health`

Health check endpoint.

## 🧪 Testing

Run the test suite:

```bash
cd backend
pytest tests/ -v --cov=app --cov-report=html
```

Run specific test files:

```bash
pytest tests/test_api.py -v
pytest tests/test_services.py -v
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek API key (required) | - |
| `DEEPSEEK_API_URL` | DeepSeek API endpoint | `https://api.deepseek.com/v1/chat/completions` |
| `QDRANT_DB_PATH` | Qdrant storage path | `./qdrant_db` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `ENABLE_CACHE` | Enable Redis caching | `true` |
| `CACHE_TTL_SECONDS` | Cache TTL in seconds | `3600` |
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `TOP_K_CHUNKS` | Number of chunks to retrieve | `5` |
| `CHUNK_SIZE` | Text chunk size (characters) | `500` |
| `CHUNK_OVERLAP` | Chunk overlap (characters) | `100` |
| `EMBEDDING_MODEL` | Sentence Transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `MAX_FILE_SIZE_MB` | Maximum file size in MB | `50` |
| `MAX_PAGES` | Maximum number of pages | `1000` |
| `MAX_CHARACTERS` | Maximum characters | `2000000` |
| `MAX_WORDS` | Maximum words | `300000` |
| `MAX_CHUNKS` | Maximum chunks | `5000` |
| `MAX_SENTENCES` | Maximum sentences | `10000` |
| `QDRANT_BATCH_SIZE` | Qdrant batch size for storing chunks | `500` |
| `EMBEDDING_BATCH_SIZE` | Batch size for embedding generation | `256` |
| `SIMILARITY_MAX_THRESHOLD` | Maximum similarity threshold | `0.20` |
| `SIMILARITY_AVG_THRESHOLD` | Average similarity threshold | `0.15` |
| `SIMILARITY_MIN_SCORE` | Minimum score for high similarity | `0.15` |
| `ENABLE_QUERY_EXPANSION` | Enable LLM query expansion | `true` |
| `SKIP_RELEVANCE_CHECK` | Skip LLM relevance check | `true` |
| `SKIP_ANSWER_VALIDATION` | Skip LLM answer validation | `true` |
| `OCR_ENABLED` | Enable OCR fallback | `true` |
| `OCR_LANGUAGES` | OCR language codes | `en` |
| `OCR_GPU` | Use GPU for OCR | `false` |
| `OCR_DPI` | DPI for OCR conversion | `300` |
| `OCR_TEXT_THRESHOLD` | Min chars before OCR fallback | `50` |

## ⚡ Performance & Streaming Processing

The application uses **streaming page-by-page processing** for optimal memory efficiency with large documents:

### Streaming Architecture
- **Page-by-Page Processing**: Documents are streamed one page at a time, never loading the entire document into memory
- **Immediate Chunking**: Each page is chunked immediately after extraction using character-based chunking (500 chars, 100 char overlap)
- **Immediate Embedding**: Chunks are embedded and stored immediately, preventing memory accumulation
- **Internal Buffering**: Qdrant storage uses internal buffering (500 chunks) for efficient batch writes while maintaining streaming semantics
- **Early Validation**: Page and chunk limits are validated incrementally, stopping processing early if limits are exceeded

### Performance Optimizations
- **Query Expansion**: LLM-powered query expansion improves retrieval accuracy by expanding user queries with related terms
- **Redis Caching**: Query results are cached to avoid redundant LLM calls (66% faster for repeated queries)
- **Optimized LLM Calls**: Reduced from 3 API calls to 1-2 calls per question (relevance check and answer validation can be skipped)
- **Lower Similarity Thresholds**: Optimized thresholds (0.20/0.15) improve recall while maintaining precision
- **Parallel Processing**: Embedding generation uses parallel batch processing for faster document ingestion

### Performance Characteristics
- **Small documents** (1-50 pages): ~20-40 seconds processing time
- **Medium documents** (50-200 pages): ~80-160 seconds processing time
- **Large documents** (200-500 pages): ~3-7 minutes processing time
- **Very large documents** (500-1000 pages): ~6-14 minutes processing time
- **Query Response Time**: 1-2 seconds per question (with caching: <100ms for cached queries)

### Memory Usage
- **Constant Memory**: Memory usage remains stable regardless of document size (only one page in memory at a time)
- **Per-Chunk Overhead**: Approximately 2KB per chunk (embeddings + text + metadata)
- **Buffer Overhead**: ~1MB for Qdrant batch buffer (500 chunks)
- **Safe for 8GB+ RAM**: Can handle 1000+ page documents without memory issues

### Algorithm Capabilities
- **PDF/DOCX/TXT Extraction**: ~0.1-0.5s per page (pdfplumber/python-docx, streamed)
- **OCR Processing**: ~2-5s per page (EasyOCR, only for scanned PDFs)
- **Text Chunking**: ~0.001s per chunk (character-based chunker)
- **Embedding Generation**: ~0.001-0.01s per chunk (parallel batch processing)
- **Qdrant Storage**: ~0.01s per batch of 500 chunks (buffered writes)
- **Query Expansion**: ~200-500ms per query (LLM call)
- **Answer Generation**: ~1-2s per answer (LLM call)

## 📊 Monitoring

The application exposes Prometheus-compatible metrics at `/metrics`:

- API request latency
- Number of queries processed
- Error rates
- LLM response times
- Token usage

Structured JSON logging is enabled by default, capturing:
- Retrieval similarity scores and metrics
- Query expansion details
- LLM response times
- Token usage statistics
- Answer lengths
- Document processing metrics
- Cache hit/miss rates
- Similarity threshold validation results

**OpenTelemetry Tracing** (optional):
- Distributed tracing for debugging
- OTLP endpoint support for production monitoring
- Console exporter for local development

## 🚢 Deployment (Render)

### Backend Deployment

1. **Create a new Web Service on Render**
   - Connect your GitHub repository
   - Build Command: `cd backend && pip install -r requirements.txt`
   - Start Command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Environment: Python 3

2. **Set Environment Variables**
   - `DEEPSEEK_API_KEY`: Your DeepSeek API key
   - `QDRANT_DB_PATH`: `/tmp/qdrant_db` (or use persistent disk)
   - `REDIS_URL`: Redis connection URL (optional but recommended)
   - Other variables as needed

3. **Deploy**

### Frontend Deployment

1. **Create a Static Site on Render**
   - Build Command: (none, or build if using a framework)
   - Publish Directory: `frontend`
   - Update `app.js` to point to your backend URL

## 🛠️ Tech Stack

### Core AI/ML
- **Python 3.10+**: Programming language
- **Sentence Transformers**: Embedding generation (all-MiniLM-L6-v2)
- **Qdrant**: Vector database (local/embedded mode)
- **DeepSeek API**: LLM for answer generation and query expansion
- **pdfplumber**: PDF text extraction
- **python-docx**: DOCX text extraction
- **EasyOCR**: OCR engine for scanned PDFs (automatic model download)
- **pdf2image**: PDF to image conversion (requires poppler system dependency)
- **Redis**: Caching layer for query results

### Backend
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation and settings
- **Prometheus Client**: Metrics collection
- **OpenTelemetry**: Distributed tracing
- **Redis**: Caching and session management

### Frontend
- **Streamlit**: Modern Python web framework for interactive UIs
- **HTML/CSS/JavaScript**: Additional static assets
- **Responsive Design**: Mobile-friendly interface

### DevOps
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **GitHub Actions**: CI/CD pipeline
- **Black & Ruff**: Code formatting and linting
- **pytest**: Testing framework

## 📝 Project Structure

```
Smart Document QA Assistant/
├── backend/
│   ├── app/
│   │   ├── api/              # API routes and schemas
│   │   │   └── routes/        # API endpoint handlers
│   │   ├── agent/            # RAG agent with query expansion
│   │   │   ├── agent.py       # Main agent orchestrator
│   │   │   ├── tools.py       # LLM, embedding, vector tools
│   │   │   ├── memory.py      # Dual memory system (Redis + Qdrant)
│   │   │   ├── prompts.py    # Prompt templates
│   │   │   └── validators.py # Question/answer validators
│   │   ├── services/          # Business logic services
│   │   │   ├── document_processor.py  # PDF/DOCX/TXT processing
│   │   │   ├── embedding_service.py  # Embedding generation
│   │   │   ├── vector_store.py       # Qdrant integration
│   │   │   ├── llm_service.py        # DeepSeek API integration
│   │   │   └── ocr_service.py        # EasyOCR integration
│   │   ├── models/            # Data models
│   │   ├── utils/            # Utility functions
│   │   └── main.py            # FastAPI application
│   ├── tests/                 # Test suite
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile             # Docker configuration
├── frontend/                  # Streamlit web interface
│   ├── app.py                 # Streamlit application
│   └── static/                # Static assets
├── docker-compose.yml         # Docker Compose configuration
├── .github/workflows/         # CI/CD workflows
└── README.md                  # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🚀 Key Features

- ✅ **Multi-Format Support**: PDF, DOCX, and TXT documents
- ✅ **OCR Support**: Automatic OCR fallback for scanned PDFs
- ✅ **Query Expansion**: LLM-powered query expansion for better retrieval
- ✅ **Multi-Document Search**: Search across all uploaded documents simultaneously
- ✅ **Redis Caching**: Fast response times for repeated queries
- ✅ **Optimized Performance**: Reduced LLM API calls (66% faster)
- ✅ **Similarity Thresholds**: Optimized thresholds for better recall
- ✅ **Streaming Processing**: Memory-efficient processing of large documents
- ✅ **Production Ready**: Monitoring, tracing, error handling, and validation
- ✅ **Web Interface**: User-friendly Streamlit frontend

## 🙏 Acknowledgments

- Sentence Transformers for embedding models
- Qdrant for vector storage
- DeepSeek for LLM API
- FastAPI for the excellent web framework
- Redis for caching capabilities
- Streamlit for the frontend framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using RAG (Retrieval-Augmented Generation)**

