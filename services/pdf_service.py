import os
import re
import urllib.parse
from typing import List, Dict, Any, Optional
import requests

try:
    import pymupdf as fitz
except ImportError:
    import fitz


# Base directories for safe file operations
DEFAULT_UPLOAD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
)
MAX_DOWNLOAD_SIZE_BYTES = int(os.getenv("MAX_CONTENT_LENGTH", 26214400))  # 25 MB default


class PDFProcessingError(Exception):
    """Base exception for PDF extraction issues."""
    pass


class PDFEncryptedError(PDFProcessingError):
    """Raised when PDF is password protected."""
    pass


class PDFCorruptedError(PDFProcessingError):
    """Raised when PDF cannot be opened or is corrupted."""
    pass


def safe_resolve_pdf_path(path_or_filename: str, allowed_dir: Optional[str] = DEFAULT_UPLOAD_DIR) -> str:
    """
    Safely resolves and validates a PDF path to prevent directory traversal.
    If allowed_dir is provided, ensures the target file resides strictly within that directory.
    """
    if not path_or_filename or not isinstance(path_or_filename, str):
        raise PDFProcessingError("Invalid file path provided.")

    if allowed_dir is not None:
        allowed_dir_abs = os.path.abspath(allowed_dir)

        # If only a filename or relative path is provided, anchor it to allowed_dir
        if not os.path.isabs(path_or_filename):
            candidate_path = os.path.normpath(os.path.join(allowed_dir_abs, path_or_filename))
        else:
            candidate_path = os.path.normpath(path_or_filename)

        # Check that candidate path is within allowed_dir
        common = os.path.commonpath([allowed_dir_abs, candidate_path])
        if common != allowed_dir_abs:
            raise PDFProcessingError("Access denied: File path is outside the allowed uploads directory.")
    else:
        candidate_path = os.path.normpath(path_or_filename)

    if not candidate_path.lower().endswith(".pdf"):
        raise PDFProcessingError("Invalid file type: File must have a .pdf extension.")

    if not os.path.isfile(candidate_path):
        raise PDFProcessingError(f"PDF file not found: {os.path.basename(candidate_path)}")

    return candidate_path


def extract_text_from_pdf(
    pdf_path: str,
    paper_id: Optional[int] = None,
    filename: Optional[str] = None,
    doi: Optional[str] = None,
    source_url: Optional[str] = None,
    allowed_dir: Optional[str] = DEFAULT_UPLOAD_DIR
) -> Dict[str, Any]:
    """
    Extracts text page by page from a PDF using PyMuPDF (fitz).
    Preserves document metadata, page numbers, and reports scanned PDFs without text.
    """
    resolved_path = safe_resolve_pdf_path(pdf_path, allowed_dir=allowed_dir)
    effective_filename = filename or os.path.basename(resolved_path)

    # Check for empty file
    if os.path.getsize(resolved_path) == 0:
        raise PDFCorruptedError("The PDF file is completely empty (0 bytes).")

    doc = None
    try:
        doc = fitz.open(resolved_path)
    except Exception as exc:
        raise PDFCorruptedError(f"Failed to open PDF (file may be damaged or invalid): {str(exc)}") from exc

    try:
        # Check password protection
        if doc.is_encrypted or doc.needs_pass:
            raise PDFEncryptedError("PDF is password-protected and cannot be extracted.")

        total_pages = len(doc)
        if total_pages == 0:
            return {
                "paper_id": paper_id,
                "filename": effective_filename,
                "filepath": resolved_path,
                "doi": doi,
                "source_url": source_url,
                "total_pages": 0,
                "has_text": False,
                "is_scanned": False,
                "extracted_text_length": 0,
                "pages": [],
                "status": "empty",
                "message": "PDF contains 0 pages."
            }

        pages_data = []
        total_text_length = 0

        for page_num in range(1, total_pages + 1):
            page = doc[page_num - 1]
            try:
                page_text = page.get_text("text") or ""
            except Exception:
                page_text = ""

            clean_page_text = page_text.strip()
            char_count = len(clean_page_text)
            total_text_length += char_count

            pages_data.append({
                "page_number": page_num,
                "text": page_text,
                "char_count": char_count,
                "has_text": char_count > 0
            })

        is_scanned = (total_text_length == 0)
        status = "scanned_no_text" if is_scanned else "success"
        
        if is_scanned:
            message = (
                "Document appears to be a scanned or image-only PDF with no extractable text. "
                "Optical Character Recognition (OCR) is not implemented in this pipeline."
            )
        else:
            message = f"Successfully extracted text from {total_pages} page(s) ({total_text_length} characters)."

        return {
            "paper_id": paper_id,
            "filename": effective_filename,
            "filepath": resolved_path,
            "doi": doi,
            "source_url": source_url,
            "total_pages": total_pages,
            "has_text": not is_scanned,
            "is_scanned": is_scanned,
            "extracted_text_length": total_text_length,
            "pages": pages_data,
            "status": status,
            "message": message
        }

    finally:
        if doc is not None:
            doc.close()


def validate_and_download_pdf(
    url: str,
    output_dir: str = DEFAULT_UPLOAD_DIR,
    filename_hint: Optional[str] = None,
    max_size_bytes: int = MAX_DOWNLOAD_SIZE_BYTES,
    timeout: int = 15
) -> Dict[str, Any]:
    """
    Validates a PDF URL and safely downloads it if under the maximum size limit.
    Enforces URL scheme checks and content-type verification.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise PDFProcessingError(f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed.")

    if not parsed.netloc:
        raise PDFProcessingError("Invalid URL: missing domain host.")

    # Determine safe filename
    if filename_hint:
        safe_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", filename_hint)
    else:
        url_path = parsed.path.rstrip("/")
        name_from_url = os.path.basename(url_path) if url_path else "downloaded_paper.pdf"
        safe_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name_from_url)

    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"

    os.makedirs(output_dir, exist_ok=True)
    destination_path = os.path.join(output_dir, safe_name)

    headers = {
        "User-Agent": "LitReviewAI/1.0 (Academic PDF Processor)"
    }

    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                raise PDFProcessingError(f"Failed to download PDF: server responded with HTTP {response.status_code}")

            # Verify Content-Type
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and "pdf" not in content_type and "octet-stream" not in content_type:
                raise PDFProcessingError(f"URL does not return a PDF (Content-Type: '{content_type}').")

            # Verify Content-Length header if present
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_size_bytes:
                raise PDFProcessingError(
                    f"PDF exceeds size limit: {int(content_length)} bytes (limit: {max_size_bytes} bytes)."
                )

            downloaded = 0
            with open(destination_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded += len(chunk)
                        if downloaded > max_size_bytes:
                            f.close()
                            if os.path.exists(destination_path):
                                os.remove(destination_path)
                            raise PDFProcessingError(
                                f"Download aborted: file exceeded maximum size of {max_size_bytes} bytes."
                            )
                        f.write(chunk)

    except requests.exceptions.RequestException as req_err:
        if os.path.exists(destination_path):
            os.remove(destination_path)
        raise PDFProcessingError(f"Network error while downloading PDF: {str(req_err)}") from req_err

    return {
        "status": "success",
        "filepath": destination_path,
        "filename": safe_name,
        "bytes_downloaded": downloaded,
        "source_url": url
    }
