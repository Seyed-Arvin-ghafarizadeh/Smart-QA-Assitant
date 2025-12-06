# Environment Setup Guide

## Problem: DEEPSEEK_API_KEY is Missing

This application requires a DeepSeek API key to function. If you see the error:
```
ValueError: DEEPSEEK_API_KEY environment variable is required
```

Follow these solutions:

---

## Solution 1: Create `.env` File (RECOMMENDED)

### Step 1: Create the `.env` file in the ROOT directory

**Windows (PowerShell):**
```powershell
# Navigate to project root
cd "D:\Portfolio\Smart Document QA Assistant"

# Create .env file
New-Item -Path ".env" -ItemType File -Force

# Add the API key (replace YOUR_ACTUAL_API_KEY with your real key)
Add-Content -Path ".env" -Value "DEEPSEEK_API_KEY=YOUR_ACTUAL_API_KEY"
Add-Content -Path ".env" -Value "DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions"
Add-Content -Path ".env" -Value "CHROMA_DB_PATH=./chroma_db"
Add-Content -Path ".env" -Value "API_HOST=0.0.0.0"
Add-Content -Path ".env" -Value "API_PORT=8000"
Add-Content -Path ".env" -Value "LOG_LEVEL=INFO"
Add-Content -Path ".env" -Value "TOP_K_CHUNKS=5"
Add-Content -Path ".env" -Value "CHUNK_SIZE=400"
Add-Content -Path ".env" -Value "CHUNK_OVERLAP=50"
Add-Content -Path ".env" -Value "EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2"
```

**Windows (Command Prompt):**
```cmd
cd "D:\Portfolio\Smart Document QA Assistant"
echo DEEPSEEK_API_KEY=YOUR_ACTUAL_API_KEY > .env
echo DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions >> .env
echo CHROMA_DB_PATH=./chroma_db >> .env
echo API_HOST=0.0.0.0 >> .env
echo API_PORT=8000 >> .env
echo LOG_LEVEL=INFO >> .env
echo TOP_K_CHUNKS=5 >> .env
echo CHUNK_SIZE=400 >> .env
echo CHUNK_OVERLAP=50 >> .env
echo EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 >> .env
```

**Linux/Mac:**
```bash
cd "Smart Document QA Assistant"
cat > .env << 'EOF'
DEEPSEEK_API_KEY=YOUR_ACTUAL_API_KEY
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
CHROMA_DB_PATH=./chroma_db
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
TOP_K_CHUNKS=5
CHUNK_SIZE=400
CHUNK_OVERLAP=50
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EOF
```

### Step 2: Edit `.env` and replace `YOUR_ACTUAL_API_KEY`

Open `.env` in any text editor (Notepad, VS Code, etc.) and replace `YOUR_ACTUAL_API_KEY` with your actual DeepSeek API key.

### Step 3: Restart the server

```powershell
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Solution 2: Set Environment Variable Directly (Quick Test)

**Windows (PowerShell) - Session Only:**
```powershell
$env:DEEPSEEK_API_KEY = "YOUR_ACTUAL_API_KEY"
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Windows (PowerShell) - Permanent:**
```powershell
[System.Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', 'YOUR_ACTUAL_API_KEY', 'User')
```
Then restart PowerShell.

**Linux/Mac:**
```bash
export DEEPSEEK_API_KEY="YOUR_ACTUAL_API_KEY"
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Solution 3: Use Python dotenv Manually

If you prefer to load `.env` from a different location:

```powershell
cd backend
python
```

```python
from dotenv import load_dotenv
import os

# Load .env from parent directory
load_dotenv(dotenv_path="../.env")
print(f"API Key loaded: {os.getenv('DEEPSEEK_API_KEY')[:10]}...")
```

---

## How to Get a DeepSeek API Key

1. Visit: https://platform.deepseek.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and use it in your `.env` file

---

## Verification

After setup, verify your configuration:

**Test 1: Check if .env file exists**
```powershell
Test-Path ".env"  # Should return True
```

**Test 2: Check if key is loaded**
```powershell
cd backend
python -c "from app.main import Settings; s = Settings(); print('API Key:', s.deepseek_api_key[:10] + '...' if s.deepseek_api_key else 'NOT FOUND')"
```

**Test 3: Start the server**
```powershell
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

If you see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```
✅ Success! Your API key is configured correctly.

---

## Troubleshooting

### Still getting the error?

1. **Check .env location**: The `.env` file should be in the ROOT directory (`D:\Portfolio\Smart Document QA Assistant\.env`), NOT in `backend/`

2. **Check .env format**: Open `.env` in a text editor and ensure:
   - No spaces around `=` sign
   - No quotes unless required
   - No extra blank lines at the start

3. **Check Python working directory**: When running uvicorn, make sure you're in the `backend/` directory:
   ```powershell
   cd backend
   pwd  # Should show: D:\Portfolio\Smart Document QA Assistant\backend
   ```

4. **Restart terminal**: After setting environment variables or creating `.env`, restart your terminal/PowerShell session

5. **Check file encoding**: Ensure `.env` is saved with UTF-8 encoding (no BOM)

### Security Note

⚠️ **NEVER** commit your `.env` file to Git! It should already be in `.gitignore`.

