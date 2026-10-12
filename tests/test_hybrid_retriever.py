import os
import unittest
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np

from services.bm25_service import BM25Service
from services.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from services.vector_store import FaissVectorStore
from agents.retrieval_agent import RetrievalAgent
from app import app


class HybridRetrieverTestCase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.chunks = [
            {
                "chunk_id": "doc1_c1",
                "document_id": "doc1",
                "filename": "paper1.pdf",
                "text": "Retrieval augmented generation enhances hallucination detection in clinical benchmarks.",
                "page_number": 1,
                "doi": "10.1000/1",
                "source_url": "https://example.com/1"
            },
            {
                "chunk_id": "doc1_c2",
                "document_id": "doc1",
                "filename": "paper1.pdf",
                "text": "Neural language models suffer from parametric memory decay over time.",
                "page_number": 2,
                "doi": "10.1000/1",
                "source_url": "https://example.com/1"
            },
            {
                "chunk_id": "doc2_c1",
                "document_id": "doc2",
                "filename": "paper2.pdf",
                "text": "Dense passage retrieval and BM25 okapi keyword search combine effectively for fact checking.",
                "page_number": 1,
                "doi": "10.1000/2",
                "source_url": "https://example.com/2"
            }
        ]

        # BM25 service
        self.bm25 = BM25Service()
        self.bm25.build_index(self.chunks)

        # Mock embedding service (4-dimensional normalized float32)
        self.mock_embed_svc = MagicMock()
        self.mock_embed_svc.dimension = 4
        self.mock_embed_svc.embed_texts.side_effect = lambda texts: np.ones((len(texts), 4), dtype=np.float32) / 2.0
        self.mock_embed_svc.embed_query.return_value = np.ones((1, 4), dtype=np.float32) / 2.0

        # Vector store
        self.store = FaissVectorStore(
            index_path=os.path.join(self.temp_dir, "faiss.bin"),
            metadata_path=os.path.join(self.temp_dir, "meta.json"),
            dimension=4
        )
        # Add chunks with dummy embeddings
        embeddings = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1, 0.1],
            [0.4, 0.4, 0.4, 0.4]
        ], dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        self.store.add_chunks(self.chunks, embeddings, doc_identifier="test_docs")

        self.hybrid_retriever = HybridRetriever(
            vector_store=self.store,
            bm25_service=self.bm25,
            embedding_service=self.mock_embed_svc,
            k=60
        )

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

    def test_bm25_search_scoring_and_ranking(self):
        """Verify BM25 Okapi matches relevant terms and ranks them properly."""
        results = self.bm25.search("clinical hallucination", top_k=2)
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertEqual(top_match["chunk_id"], "doc1_c1")
        self.assertIn("bm25_score", top_match)
        self.assertGreater(top_match["bm25_score"], 0.0)

    def test_bm25_empty_query(self):
        """Verify BM25 search returns empty list for empty query."""
        results = self.bm25.search("")
        self.assertEqual(results, [])

    def test_reciprocal_rank_fusion_math(self):
        """Verify RRF formula 1 / (k + rank) with k=60."""
        ranking_a = [
            {"chunk_id": "c1", "text": "chunk 1"},
            {"chunk_id": "c2", "text": "chunk 2"}
        ]
        ranking_b = [
            {"chunk_id": "c2", "text": "chunk 2"},
            {"chunk_id": "c3", "text": "chunk 3"}
        ]
        fused = reciprocal_rank_fusion([ranking_a, ranking_b], k=60, top_k=3)
        self.assertGreater(len(fused), 0)
        
        # c2 was rank 2 in A and rank 1 in B -> score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~= 0.016129 + 0.016393 ~= 0.03252
        # c1 was rank 1 in A only -> score = 1/(60+1) ~= 0.016393
        # c3 was rank 2 in B only -> score = 1/(60+2) ~= 0.016129
        # So c2 should be #1!
        self.assertEqual(fused[0]["chunk_id"], "c2")
        self.assertAlmostEqual(fused[0]["rrf_score"], (1.0 / 62.0) + (1.0 / 61.0), places=5)

    def test_hybrid_retrieve_with_reranking(self):
        """Verify HybridRetriever combines dense and sparse rankings with re-ranking."""
        results = self.hybrid_retriever.retrieve_hybrid("hallucination detection", top_k=2, rerank=True)
        self.assertLessEqual(len(results), 2)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertIn("passage_text", top)
        self.assertIn("relevance_score", top)
        self.assertIn("retrieval_method", top)
        self.assertEqual(top["retrieval_method"], "hybrid_rrf_reranked")

    def test_citation_based_retrieval(self):
        """Verify RetrievalAgent citation retrieval fetches referenced works without fabrication."""
        mock_openalex = MagicMock()
        mock_openalex.fetch_work_by_id.return_value = {
            "id": "https://openalex.org/W9999",
            "title": "Foundational Survey on Large Language Models",
            "authorships": [{"author": {"display_name": "Turing, Alan"}}],
            "publication_year": 2023,
            "abstract_inverted_index": None,
            "primary_location": {"landing_page_url": "https://doi.org/10.9999"},
            "doi": "https://doi.org/10.9999",
            "open_access": {"is_oa": False, "oa_url": None},
            "referenced_works": []
        }

        retriever = RetrievalAgent(openalex_service=mock_openalex)
        seed_paper = {
            "title": "Seed Paper",
            "openalex_id": "https://openalex.org/W1111",
            "referenced_works": ["https://openalex.org/W9999"]
        }

        with patch("agents.retrieval_agent.save_or_get_paper") as mock_save:
            mock_save.side_effect = lambda conn, p: p
            cited_papers = retriever.retrieve_by_citations(seed_paper, max_citations=2)
            self.assertEqual(len(cited_papers), 1)
            self.assertEqual(cited_papers[0]["title"], "Foundational Survey on Large Language Models")

    def test_hybrid_retrieval_api_endpoint(self):
        """Verify POST /api/retrieve/hybrid returns 200 with passages payload."""
        client = app.test_client()

        with patch("app.get_hybrid_retriever") as mock_get_hr:
            mock_hr = MagicMock()
            mock_hr.retrieve_hybrid.return_value = [
                {
                    "chunk_id": "c1",
                    "passage_text": "Evidence for RAG question answering.",
                    "filename": "rag.pdf",
                    "page_number": 1,
                    "relevance_score": 0.89,
                    "retrieval_method": "hybrid_rrf_reranked"
                }
            ]
            mock_get_hr.return_value = mock_hr

            response = client.post(
                "/api/retrieve/hybrid",
                json={"query": "RAG question answering", "top_k": 3}
            )
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(len(data["passages"]), 1)
            self.assertEqual(data["passages"][0]["relevance_score"], 0.89)

    def test_hybrid_retrieval_api_empty_query(self):
        """Verify POST /api/retrieve/hybrid returns 400 when query is empty."""
        client = app.test_client()
        response = client.post("/api/retrieve/hybrid", json={"query": ""})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
