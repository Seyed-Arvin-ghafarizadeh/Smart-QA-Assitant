"""PDF validation utilities for ensuring document compatibility."""
import os
import struct
from typing import Dict, List, Optional, Tuple
from fastapi import HTTPException

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None


class PDFValidator:
    """Comprehensive PDF validation for ensuring document compatibility with PyMuPDF."""
    
    @staticmethod
    def validate_pdf_header(file_path: str) -> Dict[str, any]:
        """
        Validate PDF file header and basic structure.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with validation results
            
        Raises:
            HTTPException: If PDF is invalid
        """
        try:
            with open(file_path, 'rb') as f:
                # Read first few bytes to check PDF header
                header = f.read(8)
                
                if not header.startswith(b'%PDF-'):
                    raise HTTPException(
                        status_code=400,
                        detail="File is not a valid PDF. PDF files must start with '%PDF-' header."
                    )
                
                # Extract PDF version
                try:
                    version_str = header[5:8].decode('ascii')
                    major, minor = map(int, version_str.split('.'))
                    pdf_version = f"{major}.{minor}"
                except (ValueError, IndexError):
                    pdf_version = "unknown"
                
                # Check for PDF version compatibility
                if pdf_version != "unknown":
                    if major < 1 or major > 2:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unsupported PDF version {pdf_version}. Supported: PDF 1.0-2.7"
                        )
                
                # Check file size
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                
                if file_size < 100:  # Minimum viable PDF size
                    raise HTTPException(
                        status_code=400,
                        detail=f"PDF file is too small ({file_size} bytes). Minimum size: 100 bytes."
                    )
                
                # Check for %%EOF marker
                f.seek(-1024, 2)  # Check last 1024 bytes for EOF
                end_content = f.read()
                if b'%%EOF' not in end_content:
                    raise HTTPException(
                        status_code=400,
                        detail="PDF file is corrupted or incomplete. Missing '%%EOF' marker."
                    )
                
                return {
                    "version": pdf_version,
                    "file_size": file_size,
                    "header_valid": True
                }
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to validate PDF header: {str(e)}"
            )

    @staticmethod
    def validate_pdf_structure(file_path: str) -> Dict[str, any]:
        """
        Validate PDF document structure using PyMuPDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with validation results
            
        Raises:
            HTTPException: If PDF structure is invalid
        """
        if not PYMUPDF_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="PyMuPDF is not available. PDF validation requires PyMuPDF."
            )
        
        doc = None
        try:
            # Attempt to open the PDF
            doc = fitz.open(file_path)
            
            # Check for encryption/password protection
            if doc.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="PDF is password-protected or encrypted. Please provide an unprotected PDF file."
                )
            
            # Get basic document info
            page_count = len(doc)
            if page_count == 0:
                raise HTTPException(
                    status_code=400,
                    detail="PDF contains no pages. Please provide a valid PDF with content."
                )
            
            # Validate document metadata
            metadata = doc.metadata
            if not metadata:
                # Some PDFs might not have metadata, this is not necessarily an error
                pass
            
            # Test page access and basic text extraction
            test_pages = min(3, page_count)  # Test first 3 pages
            
            pages_with_content = 0
            total_text_length = 0
            
            for page_num in range(test_pages):
                try:
                    page = doc[page_num]
                    
                    # Test basic page operations
                    if page.rect.width <= 0 or page.rect.height <= 0:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Page {page_num + 1} has invalid dimensions. PDF may be corrupted."
                        )
                    
                    # Test text extraction
                    text = page.get_text()
                    if text and len(text.strip()) > 0:
                        pages_with_content += 1
                        total_text_length += len(text)
                    
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to process page {page_num + 1}: {str(e)}. PDF may be corrupted."
                    )
            
            # Check if document has any extractable content
            if pages_with_content == 0 and page_count > 0:
                # Document might be image-only, which is acceptable with OCR
                # We'll warn but not reject
                pass
            
            # Check document permissions
            permissions = doc.permissions
            if permissions == 0:
                # No permissions might indicate issues, but not necessarily
                pass
            
            return {
                "page_count": page_count,
                "pages_with_content": pages_with_content,
                "total_text_length": total_text_length,
                "is_encrypted": False,
                "structure_valid": True,
                "metadata": metadata
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"PDF structure validation failed: {str(e)}. The PDF may be corrupted or have an unsupported format."
            )
        finally:
            if doc:
                doc.close()

    @staticmethod
    def validate_pdf_compatibility(file_path: str, max_pages: int = 1000) -> Dict[str, any]:
        """
        Comprehensive PDF compatibility validation.
        
        Args:
            file_path: Path to PDF file
            max_pages: Maximum allowed pages
            
        Returns:
            Dictionary with complete validation results
            
        Raises:
            HTTPException: If PDF is not compatible
        """
        # Step 1: Header validation
        header_info = PDFValidator.validate_pdf_header(file_path)
        
        # Step 2: Structure validation  
        structure_info = PDFValidator.validate_pdf_structure(file_path)
        
        # Step 3: Page count validation
        if structure_info["page_count"] > max_pages:
            raise HTTPException(
                status_code=400,
                detail=f"PDF has {structure_info['page_count']} pages, which exceeds the maximum of {max_pages} pages."
            )
        
        # Step 4: Content validation
        if structure_info["page_count"] > 0 and structure_info["pages_with_content"] == 0:
            # Check if it's likely an image-only PDF that would benefit from OCR
            # We'll allow this but warn that OCR may be needed
            pass
        
        # Combine results
        validation_result = {
            **header_info,
            **structure_info,
            "compatibility_score": "high",  # Assume high compatibility if we reach here
            "warnings": []
        }
        
        # Add warnings for potential issues
        if structure_info["pages_with_content"] == 0 and structure_info["page_count"] > 0:
            validation_result["warnings"].append(
                "PDF appears to contain no extractable text. This may be an image-only PDF that requires OCR processing."
            )
        
        if header_info["version"].startswith("2."):
            validation_result["warnings"].append(
                f"PDF version {header_info['version']} detected. Some features may have limited support."
            )
        
        return validation_result


def validate_uploaded_pdf(file_path: str, max_pages: int = 1000) -> Dict[str, any]:
    """
    Convenience function to validate an uploaded PDF file.
    
    Args:
        file_path: Path to uploaded PDF file
        max_pages: Maximum allowed pages
        
    Returns:
        Validation results dictionary
        
    Raises:
        HTTPException: If validation fails
    """
    return PDFValidator.validate_pdf_compatibility(file_path, max_pages)
