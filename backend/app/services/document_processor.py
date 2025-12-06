"""Document processing service for PDF extraction and chunking."""
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple

from app.models.document import Chunk, Document
from app.utils.logger import logger
from app.utils.text_cleaner import clean_text


class DocumentProcessor:
    """Handles PDF extraction, cleaning, and chunking."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        """
        Initialize document processor.

        Args:
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, List[int], int]:
        """
        Extract text from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Tuple of (full_text, page_breaks, total_pages)
        """
        try:
            full_text = ""
            page_breaks = []
            total_pages = 0

            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    page_text = clean_text(page_text)

                    if page_text:
                        # Record position where page ends
                        page_breaks.append(len(full_text))
                        full_text += page_text + "\n\n"

            logger.info(f"Extracted text from {total_pages} pages")
            return full_text, page_breaks, total_pages

        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    def chunk_text(
        self, text: str, page_breaks: List[int], document_id: str, filename: str
    ) -> List[Chunk]:
        """
        Split text into chunks with metadata.

        Args:
            text: Full document text
            page_breaks: List of character positions where pages break
            document_id: Unique document identifier
            filename: Original filename

        Returns:
            List of Chunk objects with metadata
        """
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text)

        # Determine page number for each chunk
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            # Find which page this chunk belongs to
            chunk_start = text.find(chunk_text)
            page_number = self._find_page_number(chunk_start, page_breaks)

            chunk = Chunk(
                text=chunk_text,
                page_number=page_number,
                chunk_index=idx,
                document_id=document_id,
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from document {document_id}")
        return chunks

    def _find_page_number(self, position: int, page_breaks: List[int]) -> int:
        """
        Find page number for a given character position.

        Args:
            position: Character position in text
            page_breaks: List of page break positions

        Returns:
            Page number (1-indexed)
        """
        if not page_breaks:
            return 1

        for page_num, break_pos in enumerate(page_breaks, start=1):
            if position < break_pos:
                return page_num

        return len(page_breaks) + 1

    def process_document(
        self, file_path: str, document_id: str, filename: str
    ) -> Document:
        """
        Process a PDF document: extract, clean, and chunk.

        Args:
            file_path: Path to PDF file
            document_id: Unique document identifier
            filename: Original filename

        Returns:
            Document object with processed chunks
        """
        text, page_breaks, total_pages = self.extract_text_from_pdf(file_path)
        chunks = self.chunk_text(text, page_breaks, document_id, filename)

        document = Document(
            document_id=document_id,
            filename=filename,
            total_chunks=len(chunks),
            total_pages=total_pages,
            chunks=chunks,
        )

        return document

