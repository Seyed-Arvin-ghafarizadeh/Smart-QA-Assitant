# Build stage - includes build tools for compiling Python packages
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies to a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the model during build to avoid proxy issues at runtime
# This step may take a few minutes on first build
RUN mkdir -p /app/models && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('/app/models/all-MiniLM-L6-v2')" || echo "Model download failed, will download at runtime"

# Runtime stage - minimal image without build tools
FROM python:3.10-slim

WORKDIR /app

# Install only runtime dependencies (curl for health check)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy pre-downloaded model from builder stage
COPY --from=builder /app/models /app/models

# Copy application code
COPY backend/ .

# Copy static frontend files
COPY frontend/static /app/frontend/static

# Create necessary directories
RUN mkdir -p uploads qdrant_db

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

