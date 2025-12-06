"""Text cleaning and normalization utilities."""
import re
from typing import List


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text with normalized whitespace
    """
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special control characters but keep newlines for structure
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)

    # Normalize line breaks
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\r", "\n", text)

    # Remove excessive newlines (more than 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def split_by_pages(text: str, page_breaks: List[int]) -> List[tuple[str, int]]:
    """
    Split text by page breaks and return with page numbers.

    Args:
        text: Full document text
        page_breaks: List of character positions where pages break

    Returns:
        List of (page_text, page_number) tuples
    """
    if not page_breaks:
        return [(text, 1)]

    pages = []
    start = 0

    for i, break_pos in enumerate(page_breaks):
        page_num = i + 1
        page_text = text[start:break_pos]
        if page_text.strip():
            pages.append((page_text, page_num))
        start = break_pos

    # Add remaining text as last page
    if start < len(text):
        pages.append((text[start:], len(page_breaks) + 1))

    return pages

