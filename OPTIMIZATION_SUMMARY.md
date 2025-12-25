# Optimization Summary

This document summarizes the optimizations made to reduce Docker image size and memory footprint for cloud free-tier deployment.

## Changes Implemented

### 1. Multi-Stage Docker Builds ✅

**Backend Dockerfile** (`backend/Dockerfile`):
- Converted to multi-stage build
- Build stage includes gcc/g++ for compiling Python packages
- Runtime stage excludes build tools (saves ~200-300MB)
- Uses virtual environment to reduce image size

**Frontend Dockerfile** (`frontend/Dockerfile`):
- Converted to multi-stage build
- Reduces final image size by excluding build dependencies

### 2. Dependency Optimization ✅

**Removed from `backend/requirements.txt`**:
- `transformers` (~2GB) - Not used, model loaded locally
- `langchain` (~100MB) - Not used, simple text chunking implemented
- `langchain-community` (~50MB) - Not used
- Duplicate `httpx` entry

**Kept**:
- `huggingface_hub` - Required by sentence-transformers
- All other dependencies remain

**Expected reduction**: ~500MB-1GB

### 3. Lazy Model Loading ✅

**Changes**:
- Modified `EmbeddingService` to load model on first use instead of startup
- Updated `main.py` to not pre-load model in lifespan
- Added thread-safe lazy loading with double-check locking pattern

**Benefits**:
- Faster startup time (no model loading delay)
- Lower initial memory footprint (~200-500MB saved at startup)
- Model loads automatically when first embedding is requested

**Files modified**:
- `backend/app/services/embedding_service.py`
- `backend/app/main.py`

### 4. Static Frontend Alternative ✅

**Created**:
- `frontend/static/index.html` - Lightweight HTML/JS frontend
- No Python dependencies, pure client-side JavaScript
- Served directly by FastAPI (no separate container needed)

**Benefits**:
- Eliminates need for Streamlit container (saves ~200-300MB)
- Faster page loads
- Lower memory usage
- Can be served by FastAPI or any static file server

**Integration**:
- FastAPI automatically serves static files from `/` route
- Falls back gracefully if static directory doesn't exist
- Streamlit frontend still available as alternative

## Expected Results

### Docker Image Size
- **Before**: ~2-3GB
- **After**: ~500MB-1GB
- **Reduction**: 60-70%

### Startup Memory
- **Before**: ~500MB (model loaded at startup)
- **After**: ~100-200MB (model loads on first use)
- **Reduction**: 60-80%

### Startup Time
- **Before**: ~30-60 seconds (model loading)
- **After**: ~5-10 seconds (no model loading)
- **Reduction**: ~80%

### Cloud Free Tier Compatibility
- Should now run on instances with 512MB-1GB RAM
- Faster cold starts
- Lower resource consumption

## Usage

### Option 1: Static Frontend (Recommended for Production)
The backend now serves a static HTML/JS frontend automatically:
```bash
docker-compose up backend
# Access at http://localhost:8000
```

### Option 2: Streamlit Frontend (Development)
For Streamlit frontend, use both services:
```bash
docker-compose up
# Backend: http://localhost:8000
# Frontend: http://localhost:8501
```

## Testing Recommendations

1. **Build Docker images** and check sizes:
   ```bash
   docker build -f backend/Dockerfile -t backend-optimized .
   docker images | grep backend-optimized
   ```

2. **Test startup time**:
   ```bash
   time docker-compose up backend
   ```

3. **Monitor memory usage**:
   ```bash
   docker stats smart-doc-qa-backend
   ```

4. **Test functionality**:
   - Upload a PDF document
   - Ask questions
   - Verify embeddings work (model loads on first use)

## Notes

- Model still downloads during Docker build (pre-cached)
- Model loads lazily on first API call that requires embeddings
- Static frontend is optional - Streamlit still works
- All optimizations are backward compatible

## Files Modified

1. `backend/Dockerfile` - Multi-stage build
2. `backend/requirements.txt` - Removed unused dependencies
3. `backend/app/main.py` - Lazy loading, static file serving
4. `backend/app/services/embedding_service.py` - Lazy model loading
5. `frontend/Dockerfile` - Multi-stage build
6. `frontend/static/index.html` - New static frontend (NEW FILE)

## Next Steps (Optional)

1. Consider using `python:3.10-alpine` base image (may have compatibility issues)
2. Create separate `requirements-dev.txt` for development dependencies
3. Add Docker layer caching optimization
4. Consider model quantization for further size reduction
5. Add resource limits to docker-compose.yml for free tier

