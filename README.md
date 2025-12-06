# Smart Document QA Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that enables users to upload PDF documents and ask questions about them. Built with FastAPI, Sentence Transformers, ChromaDB, and DeepSeek API.

## 🎯 Overview

This application implements a complete RAG pipeline:

1. **Document Ingestion**: Extract text from PDFs using `pdfplumber`
2. **Text Processing**: Clean and chunk documents (300-500 tokens) using LangChain
3. **Embedding Generation**: Create vector embeddings using Sentence Transformers (all-MiniLM-L6-v2)
4. **Vector Storage**: Store embeddings and metadata in ChromaDB
5. **Retrieval**: Find relevant chunks using semantic similarity search
6. **Answer Generation**: Use DeepSeek API to generate grounded answers with citations

## 🏗️ Architecture

```
Client → FastAPI → Document Processor
                   ↓
             Embedding Generator → Vector DB
                   ↓
               Query Router → Retriever → LLM
                   ↓
               Response Builder → Client
```

### Components

- **Backend**: FastAPI service with RESTful API endpoints
- **ML Pipeline**: Document processing, embeddings, vector search, LLM integration
- **Frontend**: Modern web interface for document upload and Q&A
- **Vector DB**: ChromaDB for persistent storage of embeddings
- **Monitoring**: Prometheus-compatible metrics and structured logging

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional, for containerized deployment)
- DeepSeek API key

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

4. **Configure environment variables**
   
   **IMPORTANT**: Create a `.env` file in the **root directory** (same level as `backend/` and `frontend/` folders).
   
   Copy the template below and replace `your_deepseek_api_key_here` with your actual DeepSeek API key:
   
   ```env
   # DeepSeek API Configuration (REQUIRED)
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
   
   # ChromaDB Configuration
   CHROMA_DB_PATH=./chroma_db
   
   # Backend API Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   
   # Frontend Configuration
   FRONTEND_PORT=80
   
   # Logging
   LOG_LEVEL=INFO
   
   # Retrieval Configuration
   TOP_K_CHUNKS=5
   CHUNK_SIZE=400
   CHUNK_OVERLAP=50
   
   # Embedding Model
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```
   
   > **Note**: See `SETUP_ENV.md` for detailed setup instructions.

5. **Run the backend**
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Serve the frontend**
   ```bash
   # Option 1: Using Python HTTP server
   cd frontend
   python -m http.server 8080

   # Option 2: Using nginx (if installed)
   # Configure nginx to serve frontend/ directory
   ```

7. **Access the application**
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

Upload and process a PDF document.

**Request:**
- `file`: PDF file (multipart/form-data)

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

### POST `/api/ask`

Ask a question about an uploaded document.

**Request:**
```json
{
  "document_id": "uuid-string",
  "question": "What is the main topic?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "relevant_chunks": [
    {
      "text": "Chunk text...",
      "page_number": 1,
      "similarity_score": 0.95
    }
  ],
  "confidence": 0.92,
  "token_usage": {
    "prompt_tokens": 500,
    "completion_tokens": 150
  },
  "response_time_ms": 1250.5
}
```

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
| `CHROMA_DB_PATH` | ChromaDB storage path | `./chroma_db` |
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `TOP_K_CHUNKS` | Number of chunks to retrieve | `5` |
| `CHUNK_SIZE` | Text chunk size (characters) | `400` |
| `CHUNK_OVERLAP` | Chunk overlap (characters) | `50` |
| `EMBEDDING_MODEL` | Sentence Transformers model | `sentence-transformers/all-MiniLM-L6-v2` |

## 📊 Monitoring

The application exposes Prometheus-compatible metrics at `/metrics`:

- API request latency
- Number of queries processed
- Error rates
- LLM response times
- Token usage

Structured JSON logging is enabled by default, capturing:
- Retrieval similarity scores
- LLM response times
- Token usage statistics
- Answer lengths
- Document processing metrics

## 🚢 Deployment (Render)

### Backend Deployment

1. **Create a new Web Service on Render**
   - Connect your GitHub repository
   - Build Command: `cd backend && pip install -r requirements.txt`
   - Start Command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Environment: Python 3

2. **Set Environment Variables**
   - `DEEPSEEK_API_KEY`: Your DeepSeek API key
   - `CHROMA_DB_PATH`: `/tmp/chroma_db` (or use persistent disk)
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
- **Sentence Transformers**: Embedding generation
- **ChromaDB**: Vector database
- **LangChain**: Text splitting and processing
- **DeepSeek API**: LLM for answer generation
- **pdfplumber**: PDF text extraction

### Backend
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation and settings
- **Prometheus Client**: Metrics collection

### Frontend
- **HTML/CSS/JavaScript**: Vanilla web technologies
- **Modern CSS**: Responsive design with CSS variables

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
│   │   ├── services/          # Business logic services
│   │   ├── models/            # Data models
│   │   ├── utils/            # Utility functions
│   │   └── main.py            # FastAPI application
│   ├── tests/                 # Test suite
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile             # Docker configuration
├── frontend/                  # Web interface
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

## 🙏 Acknowledgments

- Sentence Transformers for embedding models
- ChromaDB for vector storage
- DeepSeek for LLM API
- FastAPI for the excellent web framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using RAG (Retrieval-Augmented Generation)**

