"""OCR service using EasyOCR for processing scanned PDFs."""
import os
import threading
from typing import List, Optional, Tuple
from pathlib import Path

from app.utils.logger import logger

# Try to import EasyOCR and related libraries
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR is not available. OCR functionality will be disabled.")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image is not available. OCR functionality will be disabled.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow is not available. OCR functionality will be disabled.")


class EasyOCRService:
    """Service for OCR processing using EasyOCR."""

    _instance = None
    _reader = None
    _lock = threading.Lock()
    _languages = ["en"]
    _gpu = False

    def __new__(cls, languages: List[str] = None, gpu: bool = False):
        """Singleton pattern to load model only once."""
        if cls._instance is None:
            cls._instance = super(EasyOCRService, cls).__new__(cls)
            cls._languages = languages or ["en"]
            cls._gpu = gpu
        return cls._instance

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        """
        Initialize EasyOCR service.

        Args:
            languages: List of language codes (e.g., ['en', 'fr', 'es'])
            gpu: Whether to use GPU (requires CUDA)
        """
        if not EASYOCR_AVAILABLE:
            logger.warning("EasyOCR not available. OCR will be disabled.")
            return

        if languages:
            self._languages = languages
        self._gpu = gpu

        logger.info(
            f"EasyOCRService initialized (languages: {self._languages}, GPU: {self._gpu})"
        )

    def _load_reader(self):
        """Load EasyOCR reader (lazy loading, thread-safe)."""
        if not EASYOCR_AVAILABLE:
            return None

        if self._reader is None:
            with self._lock:
                if self._reader is None:
                    try:
                        logger.info(
                            f"Loading EasyOCR model for languages: {self._languages} "
                            f"(this may take a few minutes on first run, downloading ~200MB)"
                        )
                        self._reader = easyocr.Reader(
                            self._languages, gpu=self._gpu, verbose=False
                        )
                        logger.info("EasyOCR model loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to load EasyOCR model: {str(e)}", exc_info=True)
                        return None
        return self._reader

    @property
    def reader(self):
        """Get EasyOCR reader (loads lazily)."""
        return self._load_reader()

    def is_available(self) -> bool:
        """Check if OCR is available."""
        return (
            EASYOCR_AVAILABLE
            and PDF2IMAGE_AVAILABLE
            and PIL_AVAILABLE
            and self.reader is not None
        )

    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """
        Extract text from an image file using OCR.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (extracted_text, average_confidence)
        """
        if not self.is_available():
            logger.warning("OCR not available, cannot extract text from image")
            return "", 0.0

        try:
            results = self.reader.readtext(image_path)
            text_parts = []
            confidences = []

            for (bbox, text, confidence) in results:
                text_parts.append(text)
                confidences.append(confidence)

            extracted_text = "\n".join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            logger.debug(
                f"OCR extracted {len(text_parts)} text blocks from image "
                f"(avg confidence: {avg_confidence:.2%})"
            )

            return extracted_text, avg_confidence

        except Exception as e:
            logger.error(f"Error during OCR processing: {str(e)}", exc_info=True)
            return "", 0.0

    def extract_text_from_pdf_page(
        self, pdf_path: str, page_num: int, dpi: int = 300
    ) -> Tuple[str, float]:
        """
        Extract text from a specific PDF page using OCR.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            dpi: DPI for image conversion (higher = better quality but slower)

        Returns:
            Tuple of (extracted_text, average_confidence)
        """
        if not self.is_available():
            logger.warning("OCR not available, cannot extract text from PDF page")
            return "", 0.0

        try:
            # Convert PDF page to image
            images = convert_from_path(
                pdf_path, dpi=dpi, first_page=page_num, last_page=page_num
            )

            if not images:
                logger.warning(f"No image generated for PDF page {page_num}")
                return "", 0.0

            # Use first (and only) image
            image = images[0]

            # Save to temporary file for EasyOCR
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                image.save(tmp_path, "PNG")

            try:
                # Extract text using OCR
                text, confidence = self.extract_text_from_image(tmp_path)
                return text, confidence
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(
                f"Error extracting text from PDF page {page_num}: {str(e)}", exc_info=True
            )
            return "", 0.0

    def extract_text_from_pdf(
        self, pdf_path: str, page_numbers: Optional[List[int]] = None, dpi: int = 300
    ) -> List[Tuple[int, str, float]]:
        """
        Extract text from PDF pages using OCR.

        Args:
            pdf_path: Path to PDF file
            page_numbers: Optional list of page numbers to process (1-indexed)
                          If None, processes all pages
            dpi: DPI for image conversion

        Returns:
            List of (page_number, extracted_text, confidence) tuples
        """
        if not self.is_available():
            logger.warning("OCR not available, cannot extract text from PDF")
            return []

        try:
            from pdf2image import convert_from_path

            # Convert PDF pages to images
            if page_numbers:
                # Process specific pages
                images = []
                for page_num in page_numbers:
                    page_images = convert_from_path(
                        pdf_path, dpi=dpi, first_page=page_num, last_page=page_num
                    )
                    images.extend([(page_num, img) for img in page_images])
            else:
                # Process all pages
                all_images = convert_from_path(pdf_path, dpi=dpi)
                images = [(i + 1, img) for i, img in enumerate(all_images)]

            results = []
            import tempfile

            for page_num, image in images:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    image.save(tmp_path, "PNG")

                try:
                    # Extract text using OCR
                    text, confidence = self.extract_text_from_image(tmp_path)
                    results.append((page_num, text, confidence))
                    logger.info(
                        f"OCR processed page {page_num}: "
                        f"{len(text)} chars, confidence: {confidence:.2%}"
                    )
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            return results

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
            return []

