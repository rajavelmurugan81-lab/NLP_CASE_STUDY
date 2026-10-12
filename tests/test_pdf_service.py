import os
import unittest
import tempfile
import pymupdf as fitz

from services.pdf_service import (
    extract_text_from_pdf,
    safe_resolve_pdf_path,
    PDFProcessingError,
    PDFEncryptedError,
    PDFCorruptedError
)


class PDFServiceTestCase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # 1. Create a valid multi-page PDF with text
        self.valid_pdf_path = os.path.join(self.temp_dir, "test_paper.pdf")
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((50, 72), "Page 1: Introduction to Retrieval-Augmented Generation for Medicine.")
        p2 = doc.new_page()
        p2.insert_text((50, 72), "Page 2: Methods and empirical benchmarks across datasets.")
        doc.save(self.valid_pdf_path)
        doc.close()

        # 2. Create an image-only / blank scanned PDF (no text)
        self.scanned_pdf_path = os.path.join(self.temp_dir, "scanned_paper.pdf")
        doc_scanned = fitz.open()
        doc_scanned.new_page()  # Blank page with 0 text
        doc_scanned.save(self.scanned_pdf_path)
        doc_scanned.close()

        # 3. Create an empty file (0 bytes)
        self.empty_pdf_path = os.path.join(self.temp_dir, "empty_corrupt.pdf")
        with open(self.empty_pdf_path, "wb") as f:
            pass

    def tearDown(self):
        for fname in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, fname))
            except OSError:
                pass
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass

    def test_extract_text_valid_pdf(self):
        """Verify page-by-page text extraction and metadata preservation."""
        result = extract_text_from_pdf(
            pdf_path=self.valid_pdf_path,
            paper_id=101,
            filename="test_paper.pdf",
            doi="10.1234/medrag.2024",
            source_url="https://example.org/paper.pdf",
            allowed_dir=self.temp_dir
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["has_text"])
        self.assertFalse(result["is_scanned"])
        self.assertEqual(result["total_pages"], 2)
        self.assertEqual(result["paper_id"], 101)
        self.assertEqual(result["doi"], "10.1234/medrag.2024")
        self.assertEqual(len(result["pages"]), 2)
        self.assertEqual(result["pages"][0]["page_number"], 1)
        self.assertIn("Page 1: Introduction", result["pages"][0]["text"])
        self.assertEqual(result["pages"][1]["page_number"], 2)
        self.assertIn("Page 2: Methods", result["pages"][1]["text"])

    def test_extract_text_scanned_pdf(self):
        """Verify scanned image-only PDF reports no extractable text and mentions OCR."""
        result = extract_text_from_pdf(
            pdf_path=self.scanned_pdf_path,
            allowed_dir=self.temp_dir
        )

        self.assertEqual(result["status"], "scanned_no_text")
        self.assertFalse(result["has_text"])
        self.assertIn("OCR", result["message"])
        self.assertIn("not implemented", result["message"])

    def test_extract_corrupted_empty_file(self):
        """Verify 0-byte file raises PDFCorruptedError."""
        with self.assertRaises(PDFCorruptedError):
            extract_text_from_pdf(
                pdf_path=self.empty_pdf_path,
                allowed_dir=self.temp_dir
            )

    def test_path_traversal_prevention(self):
        """Verify safe_resolve_pdf_path rejects directory traversal attempts."""
        with self.assertRaises(PDFProcessingError):
            safe_resolve_pdf_path("../../etc/passwd.pdf", allowed_dir=self.temp_dir)

    def test_invalid_extension_rejected(self):
        """Verify safe_resolve_pdf_path rejects non-pdf files."""
        txt_path = os.path.join(self.temp_dir, "test.txt")
        with open(txt_path, "w") as f:
            f.write("test")

        with self.assertRaises(PDFProcessingError):
            safe_resolve_pdf_path("test.txt", allowed_dir=self.temp_dir)


if __name__ == "__main__":
    unittest.main()
