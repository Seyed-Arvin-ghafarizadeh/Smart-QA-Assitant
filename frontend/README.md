# Streamlit Frontend

Minimal and professional Streamlit frontend for the Smart Document QA Assistant.

## Features

- 📤 **Document Upload**: Upload PDF documents for processing
- ❓ **Question Answering**: Ask questions about uploaded documents
- 📊 **Answer Display**: View answers with confidence scores and response times
- 📚 **Source Citations**: See relevant document chunks with similarity scores
- 💾 **Session History**: Keep track of uploaded documents and answers

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Frontend

1. Make sure the backend server is running at `http://localhost:8000`

2. Start the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser to the URL shown in the terminal (usually `http://localhost:8501`)

## Usage

1. **Upload a Document**: Use the upload section to select and upload a PDF file
2. **Ask Questions**: Enter your question in the text area and click "Get Answer"
3. **View Results**: Answers are displayed with relevant source chunks and metadata

## Configuration

The frontend connects to the backend API at `http://localhost:8000` by default. To change this, modify the `API_BASE_URL` constant in `app.py`.

