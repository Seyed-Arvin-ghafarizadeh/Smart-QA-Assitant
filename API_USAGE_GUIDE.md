# API Usage Guide

## 🎉 Server Status: RUNNING SUCCESSFULLY ✅

Your Smart Document QA Assistant is now running without errors!

### Current Status
- ✅ Server running at `http://localhost:8000`
- ✅ Local embedding model loaded successfully
- ✅ ChromaDB initialized
- ✅ All services operational
- ✅ Health check: `http://localhost:8000/health`
- ✅ API docs: `http://localhost:8000/docs`

---

## 📝 How to Use the API

### Step 1: Upload a Document

**Endpoint:** `POST /api/upload`

**Using cURL (PowerShell):**
```powershell
$file = "path\to\your\document.pdf"
curl -X POST "http://localhost:8000/api/upload" `
  -H "accept: application/json" `
  -H "Content-Type: multipart/form-data" `
  -F "file=@$file"
```

**Using the Swagger UI:**
1. Go to http://localhost:8000/docs
2. Click on `POST /api/upload`
3. Click "Try it out"
4. Upload your PDF file
5. Click "Execute"

**Response Example:**
```json
{
  "document_id": "8bdfa4c9-1afe-44ca-a048-9661a6e45276",
  "filename": "sample.pdf",
  "total_chunks": 15,
  "total_pages": 6,
  "message": "Document uploaded and processed successfully"
}
```

**⚠️ IMPORTANT:** Save the `document_id` from the response! You'll need it to ask questions.

---

### Step 2: Ask Questions

**Endpoint:** `POST /api/ask`

**Request Body:**
```json
{
  "document_id": "8bdfa4c9-1afe-44ca-a048-9661a6e45276",  // Use the ID from upload response
  "question": "What is the main topic of this document?",
  "top_k": 5  // Optional, defaults to 5
}
```

**Using cURL (PowerShell):**
```powershell
$body = @{
    document_id = "8bdfa4c9-1afe-44ca-a048-9661a6e45276"
    question = "What is the main topic?"
    top_k = 5
} | ConvertTo-Json

curl -X POST "http://localhost:8000/api/ask" `
  -H "accept: application/json" `
  -H "Content-Type: application/json" `
  -d $body
```

**Using the Swagger UI:**
1. Go to http://localhost:8000/docs
2. Click on `POST /api/ask`
3. Click "Try it out"
4. Enter your `document_id`, `question`, and optionally `top_k`
5. Click "Execute"

**Response Example:**
```json
{
  "answer": "The main topic of the document is...",
  "relevant_chunks": [
    {
      "text": "Chunk text here...",
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

---

## 🐛 Common Issues & Solutions

### Issue 1: "Document not found" Error ❌

**Error Message:**
```
Error answering question: Document one not found
```

**Cause:** Using an invalid or incorrect `document_id`.

**Solution:**
- ✅ Use the **exact UUID** returned from the upload response
- ❌ Don't use descriptive names like "one", "my document", "sample"

**Example:**
```json
// ❌ WRONG
{
  "document_id": "one",
  "question": "What is this about?"
}

// ✅ CORRECT
{
  "document_id": "8bdfa4c9-1afe-44ca-a048-9661a6e45276",
  "question": "What is this about?"
}
```

---

### Issue 2: Upload Succeeds but Returns 500 Error

**Status:** ✅ FIXED

**What happened:** There was a logging conflict with the `filename` field in LogRecord.

**Fix applied:** Changed `filename` to `doc_filename` in the logging extra data.

---

### Issue 3: Model Download Errors / Proxy Issues

**Status:** ✅ FIXED

**What happened:** Server tried to download the model from HuggingFace and encountered proxy/SSL errors.

**Fix applied:** 
- Modified `embedding_service.py` to automatically detect locally downloaded models
- Now uses the model in `backend/models/sentence-transformers_all-MiniLM-L6-v2/`

---

### Issue 4: DEEPSEEK_API_KEY Missing

**Status:** ✅ FIXED (Partially - You still need to add your key)

**What happened:** The `.env` file didn't exist.

**Fix applied:**
1. Created `.env` file in root directory
2. Updated Settings class to load from parent directory
3. Added comprehensive setup guide in `SETUP_ENV.md`

**⚠️ ACTION REQUIRED:**
You still need to add your actual DeepSeek API key to the `.env` file. Without it, the `/api/ask` endpoint will fail.

**How to add your key:**
```powershell
# Edit .env file and replace 'your_deepseek_api_key_here' with your actual key
(Get-Content .env) -replace 'your_deepseek_api_key_here', 'YOUR_ACTUAL_KEY' | Set-Content .env
```

Or manually edit `.env` in any text editor.

---

## 📊 Monitoring & Metrics

### Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Smart Document QA Assistant"
}
```

### Metrics (Prometheus Format)
```bash
curl http://localhost:8000/api/metrics
```

---

## 🔍 Debugging Tips

### 1. Check Server Logs

Look for these success indicators:
```
✅ "Found local model at: D:\\Portfolio\\Smart Document QA Assistant\\backend\\models\\sentence-transformers_all-MiniLM-L6-v2"
✅ "Embedding model loaded successfully from local path"
✅ "ChromaDB initialized at ./chroma_db"
✅ "All services initialized successfully"
✅ "Application startup complete"
```

### 2. View Available Documents

Unfortunately, there's no "list documents" endpoint yet. Best practice:
- **Save document IDs** after upload
- Consider implementing a simple database or JSON file to track uploads
- Or add a new endpoint to list all collections in ChromaDB

### 3. Check ChromaDB Collections

```python
# Python script to list documents
import chromadb

client = chromadb.PersistentClient(path="./backend/chroma_db")
collections = client.list_collections()

print("Available documents:")
for col in collections:
    # Collection names are in format "doc_<uuid>"
    doc_id = col.name.replace("doc_", "")
    print(f"  - {doc_id} ({col.count()} chunks)")
```

---

## 🚀 Quick Test Workflow

### Test 1: Upload a Document
```powershell
# Upload
curl -X POST "http://localhost:8000/api/upload" `
  -F "file=@sample.pdf"

# Save the document_id from the response
```

### Test 2: Ask a Question
```powershell
# Ask (replace <DOCUMENT_ID> with actual ID)
$body = @{
    document_id = "<DOCUMENT_ID>"
    question = "What is this document about?"
} | ConvertTo-Json

curl -X POST "http://localhost:8000/api/ask" `
  -H "Content-Type: application/json" `
  -d $body
```

### Test 3: Check Metrics
```powershell
curl http://localhost:8000/api/metrics
```

---

## 📚 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload and process a PDF document |
| POST | `/api/ask` | Ask a question about a document |
| GET | `/api/metrics` | Get Prometheus metrics |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive API documentation (Swagger UI) |
| GET | `/openapi.json` | OpenAPI schema |

### Request/Response Schemas

See the full schemas at: http://localhost:8000/docs

---

## 🎯 Next Steps

1. **Add Your DeepSeek API Key** to `.env` file (REQUIRED for asking questions)
2. **Test the API** using the Quick Test Workflow above
3. **Serve the Frontend** to get a nice UI:
   ```powershell
   cd frontend
   python -m http.server 8080
   ```
   Then open http://localhost:8080

---

## 💡 Tips for Production

1. **Document ID Management**: Implement a database to track uploaded documents
2. **File Cleanup**: Set up periodic cleanup of the `uploads/` directory
3. **Error Handling**: Add retry logic for LLM API calls
4. **Rate Limiting**: Add rate limiting to prevent abuse
5. **Authentication**: Add API key authentication for production use
6. **CORS**: Review CORS settings in `main.py` for your frontend domain
7. **Environment Variables**: Never commit `.env` to Git
8. **Monitoring**: Set up Prometheus to scrape `/api/metrics`

---

## 📧 Support

If you encounter any issues:
1. Check the terminal logs for error messages
2. Verify your `.env` file is correctly configured
3. Ensure you're using the correct document IDs (UUIDs)
4. Check the API documentation at http://localhost:8000/docs

---

**Last Updated:** 2025-12-06
**Status:** All systems operational ✅

