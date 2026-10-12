import io
import os
import unittest
from unittest.mock import patch, MagicMock
from app import app


class RagAPITestCase(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_upload_missing_files(self):
        """Verify POST /api/upload returns 400 when no files are provided."""
        response = self.client.post("/api/upload", data={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")

    def test_upload_valid_and_invalid_extension(self):
        """Verify upload saves valid PDFs and rejects non-PDF extensions."""
        data = {
            "files": [
                (io.BytesIO(b"%PDF-1.4 test content"), "sample_paper.pdf"),
                (io.BytesIO(b"malicious script"), "script.exe")
            ]
        }
        response = self.client.post(
            "/api/upload",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 201)
        res_data = response.get_json()
        self.assertEqual(res_data.get("status"), "success")
        self.assertEqual(res_data.get("count"), 1)
        self.assertEqual(res_data["files"][0]["filename"], "sample_paper.pdf")
        self.assertTrue(len(res_data.get("errors", [])) > 0)

        # Cleanup uploaded test file if created
        test_path = os.path.join(app.config["UPLOAD_FOLDER"], "sample_paper.pdf")
        if os.path.exists(test_path):
            try:
                os.remove(test_path)
            except OSError:
                pass

    def test_ingest_path_traversal_rejection(self):
        """Verify POST /api/ingest returns 400 for path traversal attempts."""
        response = self.client.post(
            "/api/ingest",
            json={"filename": "../../windows/system32/cmd.pdf"}
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("Access denied", data.get("message"))

    @patch("app.get_rag_agent")
    def test_ingest_success(self, mock_get_agent):
        """Verify POST /api/ingest returns 200 on successful document ingestion."""
        # Create a dummy PDF in uploads folder for testing
        test_filename = "dummy_for_ingest.pdf"
        dummy_path = os.path.join(app.config["UPLOAD_FOLDER"], test_filename)
        with open(dummy_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy")

        try:
            mock_agent = MagicMock()
            mock_agent.ingest_pdf.return_value = {
                "status": "success",
                "chunks_indexed": 4,
                "total_pages": 1,
                "message": "Indexed successfully."
            }
            mock_get_agent.return_value = mock_agent

            response = self.client.post("/api/ingest", json={"filename": test_filename})
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data.get("status"), "success")
            self.assertEqual(data.get("chunks_indexed"), 4)
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    def test_retrieve_missing_query(self):
        """Verify POST /api/retrieve returns 400 when query is empty."""
        response = self.client.post("/api/retrieve", json={"query": ""})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")

    @patch("app.get_rag_agent")
    def test_retrieve_success(self, mock_get_agent):
        """Verify POST /api/retrieve returns 200 with passage evidence list."""
        mock_agent = MagicMock()
        mock_agent.retrieve_passages.return_value = {
            "status": "success",
            "query": "Clinical question answering",
            "count": 1,
            "passages": [
                {
                    "chunk_id": "c1",
                    "passage_text": "Medical RAG benchmarks showed 18% improvement.",
                    "relevance_score": 0.88,
                    "paper_id": 1,
                    "filename": "medrag.pdf",
                    "page_number": 1
                }
            ],
            "message": "Retrieved 1 passage."
        }
        mock_get_agent.return_value = mock_agent

        response = self.client.post(
            "/api/retrieve",
            json={"query": "Clinical question answering", "top_k": 3}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(len(data.get("passages")), 1)

    @patch("app.get_rag_agent")
    def test_rag_stats(self, mock_get_agent):
        """Verify GET /api/rag/stats returns 200 and store statistics."""
        mock_agent = MagicMock()
        mock_agent.vector_store.get_stats.return_value = {
            "total_vectors": 12,
            "total_chunks": 12,
            "ingested_documents_count": 2
        }
        mock_get_agent.return_value = mock_agent

        response = self.client.get("/api/rag/stats")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data["stats"]["total_vectors"], 12)


if __name__ == "__main__":
    unittest.main()
