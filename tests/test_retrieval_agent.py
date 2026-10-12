import os
import unittest
from unittest.mock import patch

from agents.retrieval_agent import RetrievalAgent
from database.db import init_db, get_db_connection


class RetrievalAgentTestCase(unittest.TestCase):

    def setUp(self):
        # Use an isolated test database for retrieval agent tests
        self.test_db_path = "database/test_retrieval.db"
        os.environ["DATABASE_PATH"] = self.test_db_path
        init_db()
        self.agent = RetrievalAgent(default_limit=5)

    def tearDown(self):
        # Reset database env and clean up test file
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except OSError:
                pass
        os.environ["DATABASE_PATH"] = "database/literature_review.db"

    def test_normalize_work_full_metadata(self):
        """Verify normalizer maps full OpenAlex work record accurately."""
        raw_work = {
            "id": "https://openalex.org/W99999",
            "title": "Agentic Workflows in Scientific Discovery",
            "publication_year": 2024,
            "doi": "https://doi.org/10.1234/test.2024",
            "authorships": [
                {"author": {"display_name": "Alice Smith"}},
                {"author": {"display_name": "Bob Jones"}}
            ],
            "abstract_inverted_index": {
                "Scientific": [0],
                "workflows": [1],
                "accelerate": [2],
                "research.": [3]
            },
            "primary_location": {
                "landing_page_url": "https://publisher.org/article/99999",
                "pdf_url": "https://publisher.org/article/99999.pdf"
            },
            "open_access": {
                "is_oa": True,
                "oa_url": "https://publisher.org/article/99999.pdf"
            }
        }

        normalized = self.agent.normalize_work(raw_work)
        self.assertEqual(normalized["openalex_id"], "https://openalex.org/W99999")
        self.assertEqual(normalized["title"], "Agentic Workflows in Scientific Discovery")
        self.assertEqual(normalized["authors"], "Alice Smith, Bob Jones")
        self.assertEqual(normalized["published_year"], 2024)
        self.assertEqual(normalized["doi"], "https://doi.org/10.1234/test.2024")
        self.assertEqual(normalized["abstract"], "Scientific workflows accelerate research.")
        self.assertEqual(normalized["url"], "https://publisher.org/article/99999")
        self.assertEqual(normalized["pdf_url"], "https://publisher.org/article/99999.pdf")
        self.assertTrue(normalized["is_oa"])

    def test_normalize_work_missing_optional_fields(self):
        """Verify normalizer handles missing abstract, pdf, and authors safely."""
        raw_work = {
            "id": "https://openalex.org/W88888",
            "display_name": "Paywalled Paper Without Abstract",
            "authorships": [],
            "open_access": {"is_oa": False}
        }

        normalized = self.agent.normalize_work(raw_work)
        self.assertEqual(normalized["openalex_id"], "https://openalex.org/W88888")
        self.assertEqual(normalized["title"], "Paywalled Paper Without Abstract")
        self.assertEqual(normalized["authors"], "Unknown Authors")
        self.assertIsNone(normalized["abstract"])
        self.assertIsNone(normalized["pdf_url"])
        self.assertFalse(normalized["is_oa"])

    @patch("agents.retrieval_agent.search_works")
    def test_retrieve_papers_deduplication(self, mock_search):
        """Verify that identical papers retrieved across searches are deduplicated in SQLite."""
        mock_work = {
            "id": "https://openalex.org/W77777",
            "title": "Deduplication Study in Vector Search",
            "publication_year": 2023,
            "doi": "https://doi.org/10.1111/dedup.2023",
            "authorships": [{"author": {"display_name": "Researcher One"}}],
            "abstract_inverted_index": {"Study": [0], "on": [1], "dedup.": [2]},
            "primary_location": {"landing_page_url": "https://example.com/p1"}
        }
        mock_search.return_value = [mock_work]

        # First query execution
        result_1 = self.agent.retrieve_papers("deduplication in search", limit=5, save_to_db=True)
        self.assertEqual(result_1["count"], 1)
        paper_id_1 = result_1["papers"][0]["id"]

        # Second query execution returning the same paper
        result_2 = self.agent.retrieve_papers("vector deduplication", limit=5, save_to_db=True)
        self.assertEqual(result_2["count"], 1)
        paper_id_2 = result_2["papers"][0]["id"]

        # IDs must match, confirming deduplication
        self.assertEqual(paper_id_1, paper_id_2)

        # Confirm database has only 1 paper record
        conn = get_db_connection()
        count = conn.execute("SELECT COUNT(*) FROM papers WHERE openalex_id = 'https://openalex.org/W77777'").fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
