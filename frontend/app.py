"""Streamlit frontend for Smart Document QA Assistant."""
import streamlit as st
import requests
from typing import Optional
import time

# Page configuration
st.set_page_config(
    page_title="Smart Document QA Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API Configuration
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE_URL = f"{BACKEND_URL}/api"
HEALTH_URL = f"{BACKEND_URL}/health"

# Custom CSS for minimal, professional design
st.markdown("""
    <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom styling */
        .main-header {
            text-align: center;
            padding: 2rem 0;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 2rem;
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            color: #1f2937;
            margin-bottom: 0.5rem;
        }
        
        .main-header p {
            color: #6b7280;
            font-size: 1.1rem;
        }
        
        .upload-section {
            background: #f9fafb;
            padding: 2rem;
            border-radius: 12px;
            border: 2px dashed #d1d5db;
            margin-bottom: 2rem;
        }
        
        .answer-box {
            background: #f0f9ff;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
            margin: 1rem 0;
        }
        
        .chunk-box {
            background: #ffffff;
            padding: 1rem;
            border-radius: 6px;
            border-left: 3px solid #60a5fa;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .metric-box {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: #e0f2fe;
            border-radius: 6px;
            margin-right: 1rem;
            font-size: 0.9rem;
        }
        
        .stButton>button {
            width: 100%;
            background-color: #3b82f6;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.75rem;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #2563eb;
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []
if "answers" not in st.session_state:
    st.session_state.answers = []


def check_backend_health() -> bool:
    """Check if backend is running."""
    try:
        response = requests.get(HEALTH_URL, timeout=2)
        return response.status_code == 200
    except:
        return False


def upload_document(file) -> Optional[dict]:
    """Upload a PDF document to the backend."""
    try:
        files = {"file": (file.name, file.read(), "application/pdf")}
        # Increased timeout to 300 seconds (5 minutes) for large documents
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=300)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None


def ask_question(question: str) -> Optional[dict]:
    """Ask a question to the backend."""
    try:
        payload = {"question": question}
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            if isinstance(error_detail, list):
                error_detail = error_detail[0].get('msg', 'Validation error')
            st.error(f"Error: {error_detail}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None


# Main UI
st.markdown("""
    <div class="main-header">
        <h1>📄 Smart Document QA Assistant</h1>
        <p>Upload PDF, DOCX, or TXT documents and ask questions powered by RAG</p>
    </div>
""", unsafe_allow_html=True)

# Check backend health
if not check_backend_health():
    st.error("⚠️ Backend server is not running. Please start the backend server at http://localhost:8000")
    st.stop()

# Sidebar for settings (optional)
with st.sidebar:
    st.header("Settings")
    st.info("Backend: Connected ✅")
    
    # OCR status indicator
    st.markdown("---")
    st.subheader("OCR Status")
    try:
        # Try to get OCR status from backend (if endpoint exists)
        health_response = requests.get(HEALTH_URL, timeout=2)
        if health_response.status_code == 200:
            st.success("OCR: Available ✅")
            st.caption("EasyOCR is ready for scanned PDFs")
        else:
            st.warning("OCR: Status unknown")
    except:
        st.info("OCR: Check backend connection")
    
    st.markdown("---")
    if st.button("Clear History"):
        st.session_state.uploaded_documents = []
        st.session_state.answers = []
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a document file",
        type=["pdf", "docx", "txt"],
        help="Upload a PDF, DOCX, or TXT document to analyze"
    )
    
    if uploaded_file is not None:
        if st.button("Upload & Process", type="primary"):
            with st.spinner("Processing document..."):
                result = upload_document(uploaded_file)
                
                if result:
                    st.success(f"✅ Document uploaded successfully!")
                    
                    # Display upload information
                    upload_info = {
                        "Filename": result.get("filename"),
                        "Total Chunks": result.get("total_chunks"),
                        "Total Pages": result.get("total_pages"),
                        "Document ID": result.get("document_id")
                    }
                    
                    # Add OCR information if available
                    if result.get("ocr_used"):
                        upload_info["OCR Processing"] = "Yes"
                        upload_info["OCR Pages"] = result.get("ocr_pages_count", 0)
                    
                    st.json(upload_info)
                    
                    # Show OCR status if applicable
                    if result.get("ocr_used"):
                        st.info("📸 OCR was used to process scanned/image-based pages in this document.")
                    
                    # Store in session state
                    st.session_state.uploaded_documents.append(result)

with col2:
    st.header("❓ Ask Questions")
    
    # Check if documents are uploaded
    has_documents = len(st.session_state.uploaded_documents) > 0
    
    if not has_documents:
        st.info("📄 Please upload a document first to ask questions.")
    
    question = st.text_area(
        "Enter your question",
        height=100,
        placeholder="What would you like to know about the uploaded documents?",
        help="Ask questions about any uploaded documents",
        key="question_input"
    )
    
    # Enable button if question is entered AND documents are uploaded
    button_disabled = not question.strip() or not has_documents
    
    if st.button("Get Answer", type="primary", disabled=button_disabled):
        if not question.strip():
            st.warning("Please enter a question")
        elif not has_documents:
            st.warning("Please upload a document first")
        else:
            with st.spinner("Generating answer..."):
                result = ask_question(question.strip())
                
                if result:
                    # Store answer in session state
                    answer_data = {
                        "question": question.strip(),
                        "answer": result.get("answer"),
                        "confidence": result.get("confidence"),
                        "response_time_ms": result.get("response_time_ms"),
                        "relevant_chunks": result.get("relevant_chunks", []),
                        "token_usage": result.get("token_usage")
                    }
                    st.session_state.answers.append(answer_data)
                    st.rerun()

# Display answers
if st.session_state.answers:
    st.divider()
    st.header("💬 Answers")
    
    # Show most recent answer first
    for idx, answer_data in enumerate(reversed(st.session_state.answers)):
        with st.expander(f"Q: {answer_data['question']}", expanded=(idx == 0)):
            # Display answer
            st.markdown(f"<div class='answer-box'>{answer_data['answer']}</div>", unsafe_allow_html=True)
            
            # Display page numbers and source information
            if answer_data.get('relevant_chunks'):
                st.markdown("---")
                st.subheader("📄 Source Pages")
                
                # Extract unique page numbers from relevant chunks
                page_numbers = []
                for chunk in answer_data['relevant_chunks']:
                    page_num = chunk.get('page_number', 0)
                    if page_num > 0 and page_num not in page_numbers:
                        page_numbers.append(page_num)
                
                # Sort page numbers
                page_numbers.sort()
                
                if page_numbers:
                    # Display page numbers in a nice format
                    if len(page_numbers) == 1:
                        page_info = f"**Page {page_numbers[0]}**"
                    elif len(page_numbers) <= 5:
                        page_info = f"**Pages:** {', '.join(map(str, page_numbers))}"
                    else:
                        page_info = f"**Pages:** {', '.join(map(str, page_numbers[:5]))} and {len(page_numbers) - 5} more"
                    
                    st.markdown(f"<div style='background: #e0f2fe; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>{page_info}</div>", unsafe_allow_html=True)
                    
                    # Show detailed chunk information in expandable sections
                    with st.expander("View Source Chunks", expanded=False):
                        for chunk_idx, chunk in enumerate(answer_data['relevant_chunks'], 1):
                            page_num = chunk.get('page_number', 0)
                            chapter_num = chunk.get('chapter_number')
                            similarity = chunk.get('similarity_score', 0)
                            
                            chunk_header = f"**Chunk {chunk_idx}** - Page {page_num}"
                            if chapter_num:
                                chunk_header += f", Chapter {chapter_num}"
                            chunk_header += f" (Similarity: {similarity:.2%})"
                            
                            st.markdown(chunk_header)
                            st.markdown(f"<div class='chunk-box'>{chunk.get('text', '')[:500]}...</div>", unsafe_allow_html=True)
                            st.markdown("---")

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>"
    "Powered by RAG (Retrieval-Augmented Generation) | "
    "Built with Streamlit & FastAPI"
    "</div>",
    unsafe_allow_html=True
)

