import unittest
from unittest.mock import patch

from app import app
from services.openalex_service import OpenAlexConnectionError, OpenAlexAPIError


class SearchAPITestCase(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_search_missing_topic(self):
        """Verify POST /api/search returns 400 when topic is empty."""
        response = self.client.post("/api/search", json={"topic": ""})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")

    @patch("app.RetrievalAgent.retrieve_papers")
    def test_search_successful_retrieval(self, mock_retrieve):
        """Verify POST /api/search returns 200 with papers payload when retrieval succeeds."""
        mock_retrieve.return_value = {
            "status": "success",
            "query_id": 1,
            "topic": "Neural Machine Translation",
            "count": 1,
            "papers": [
                {
                    "id": 1,
                    "title": "Attention Is All You Need",
                    "authors": "Vaswani et al.",
                    "published_year": 2017,
                    "abstract": "The dominant sequence transduction models...",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
                    "openalex_id": "https://openalex.org/W2964312345",
                    "doi": None,
                    "is_oa": True
                }
            ],
            "message": "Successfully retrieved 1 paper(s) from OpenAlex."
        }

        response = self.client.post("/api/search", json={"topic": "Neural Machine Translation", "limit": 5})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("count"), 1)
        self.assertEqual(data["papers"][0]["title"], "Attention Is All You Need")

    @patch("app.RetrievalAgent.retrieve_papers")
    def test_search_connection_error(self, mock_retrieve):
        """Verify POST /api/search returns 502 when OpenAlex connection fails."""
        mock_retrieve.side_effect = OpenAlexConnectionError("Connection timeout")

        response = self.client.post("/api/search", json={"topic": "Transformers"})
        self.assertEqual(response.status_code, 502)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("Unable to reach OpenAlex", data.get("message"))

    @patch("app.RetrievalAgent.retrieve_papers")
    def test_search_api_error(self, mock_retrieve):
        """Verify POST /api/search returns 502 when OpenAlex returns HTTP error."""
        mock_retrieve.side_effect = OpenAlexAPIError("HTTP 500: Server Error")

        response = self.client.post("/api/search", json={"topic": "Transformers"})
        self.assertEqual(response.status_code, 502)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("OpenAlex API error", data.get("message"))


if __name__ == "__main__":
    unittest.main()
