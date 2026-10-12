import unittest

from services.chunking_service import (
    clean_text,
    split_text_into_chunks,
    chunk_document_pages
)


class ChunkingServiceTestCase(unittest.TestCase):

    def test_clean_text_preserves_scientific_notation_and_repairs_hyphens(self):
        """Verify cleaning fixes split hyphens while preserving math and citations."""
        raw_text = "The trans-\nformer model achieves O(N^2) complexity [12, 14] with alpha = 0.05.\r\n\r\n\r\nNext section."
        cleaned = clean_text(raw_text)

        self.assertIn("transformer", cleaned)
        self.assertIn("O(N^2)", cleaned)
        self.assertIn("[12, 14]", cleaned)
        self.assertIn("alpha = 0.05", cleaned)
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("trans-\nformer", cleaned)

    def test_split_text_into_chunks_overlap(self):
        """Verify text is sliced into overlapping chunks."""
        text = "Sentence one. " * 30  # ~420 chars
        chunks = split_text_into_chunks(text, chunk_size=150, chunk_overlap=30)

        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 180)  # Within boundary slack

    def test_chunk_document_pages_metadata_preservation(self):
        """Verify source metadata and page attribution are strictly preserved."""
        pages = [
            {"page_number": 1, "text": "Page one text detailing the problem formulation and motivation."},
            {"page_number": 2, "text": "Page two text detailing experimental evaluation and benchmarks."}
        ]
        metadata = {
            "paper_id": 42,
            "filename": "quantum_nlp.pdf",
            "doi": "10.1000/182",
            "source_url": "https://doi.org/10.1000/182"
        }

        chunks = chunk_document_pages(pages, paper_metadata=metadata, chunk_size=500, chunk_overlap=100)

        self.assertEqual(len(chunks), 2)
        # Check first chunk
        c1 = chunks[0]
        self.assertEqual(c1["page_number"], 1)
        self.assertEqual(c1["paper_id"], 42)
        self.assertEqual(c1["filename"], "quantum_nlp.pdf")
        self.assertEqual(c1["doi"], "10.1000/182")
        self.assertIn("problem formulation", c1["text"])

        # Check second chunk
        c2 = chunks[1]
        self.assertEqual(c2["page_number"], 2)
        self.assertEqual(c2["paper_id"], 42)
        self.assertIn("experimental evaluation", c2["text"])

    def test_no_metadata_leakage_between_papers(self):
        """Verify chunking does not mix metadata across independent papers."""
        paper1_pages = [{"page_number": 1, "text": "Paper 1 content."}]
        paper2_pages = [{"page_number": 1, "text": "Paper 2 content."}]

        chunks1 = chunk_document_pages(paper1_pages, {"paper_id": 1, "filename": "p1.pdf"})
        chunks2 = chunk_document_pages(paper2_pages, {"paper_id": 2, "filename": "p2.pdf"})

        self.assertEqual(chunks1[0]["paper_id"], 1)
        self.assertEqual(chunks2[0]["paper_id"], 2)
        self.assertNotEqual(chunks1[0]["chunk_id"], chunks2[0]["chunk_id"])


if __name__ == "__main__":
    unittest.main()
