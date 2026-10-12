import unittest
from unittest.mock import patch, MagicMock
import requests

from services.openalex_service import (
    reconstruct_abstract,
    search_works,
    OpenAlexConnectionError,
    OpenAlexAPIError
)


class OpenAlexServiceTestCase(unittest.TestCase):

    def test_reconstruct_abstract_success(self):
        """Verify inverted index is properly restored to linear text."""
        inverted_index = {
            "Large": [0],
            "language": [1],
            "models": [2],
            "enhance": [3],
            "NLP": [4],
            "benchmarks.": [5]
        }
        reconstructed = reconstruct_abstract(inverted_index)
        self.assertEqual(reconstructed, "Large language models enhance NLP benchmarks.")

    def test_reconstruct_abstract_empty_or_none(self):
        """Verify None or empty inverted index returns None."""
        self.assertIsNone(reconstruct_abstract(None))
        self.assertIsNone(reconstruct_abstract({}))
        self.assertIsNone(reconstruct_abstract("invalid"))

    @patch("services.openalex_service.requests.get")
    def test_search_works_success(self, mock_get):
        """Verify successful API call returns raw work records."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/W12345",
                    "title": "Retrieval Augmented Generation in Medicine",
                    "publication_year": 2024
                }
            ]
        }
        mock_get.return_value = mock_response

        results = search_works(query="RAG Medicine", limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "https://openalex.org/W12345")
        mock_get.assert_called_once()
        # Verify query parameters
        call_kwargs = mock_get.call_args[1]
        self.assertEqual(call_kwargs["params"]["search"], "RAG Medicine")
        self.assertEqual(call_kwargs["params"]["per_page"], 5)

    def test_search_works_empty_query(self):
        """Verify empty query returns empty list without network call."""
        results = search_works(query="   ", limit=10)
        self.assertEqual(results, [])

    @patch("services.openalex_service.requests.get")
    def test_search_works_timeout(self, mock_get):
        """Verify network timeout raises OpenAlexConnectionError."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        with self.assertRaises(OpenAlexConnectionError):
            search_works(query="Machine Learning")

    @patch("services.openalex_service.requests.get")
    def test_search_works_api_error(self, mock_get):
        """Verify HTTP 500 status raises OpenAlexAPIError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        with self.assertRaises(OpenAlexAPIError):
            search_works(query="Deep Learning")


if __name__ == "__main__":
    unittest.main()
