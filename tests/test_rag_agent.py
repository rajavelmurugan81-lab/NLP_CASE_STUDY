import os
import unittest
import tempfile
from unittest.mock import MagicMock
import numpy as np
import pymupdf as fitz

from agents.rag_agent import RagAgent
from services.vector_store import FaissVectorStore


class RagAgentTestCase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pdf_path = os.path.join(self.temp_dir, "sample.pdf")

        # Create synthetic PDF
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((50, 72), "Evaluation benchmarks for medical information retrieval show strong improvements with RAG.")
        doc.save(self.pdf_path)
        doc.close()

        # Mock embedding service (offline, 4-dim vectors)
        self.mock_embed_svc = MagicMock()
        self.mock_embed_svc.dimension = 4
        self.mock_embed_svc.embed_texts.side_effect = lambda texts: np.ones((len(texts), 4), dtype=np.float32) / 2.0
        self.mock_embed_svc.embed_query.return_value = np.ones((1, 4), dtype=np.float32) / 2.0

        # Vector store in temp dir
        self.store = FaissVectorStore(
            index_path=os.path.join(self.temp_dir, "faiss.bin"),
            metadata_path=os.path.join(self.temp_dir, "meta.json"),
            dimension=4
        )

        self.agent = RagAgent(
            embedding_service=self.mock_embed_svc,
            vector_store=self.store
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

    def test_retrieve_from_empty_knowledge_base(self):
        """Verify querying empty knowledge base returns explicit status."""
        result = self.agent.retrieve_passages("medical benchmarks")
        self.assertEqual(result["status"], "empty_knowledge_base")
        self.assertEqual(result["count"], 0)
        self.assertIn("empty", result["message"].lower())

    def test_end_to_end_ingestion_and_retrieval(self):
        """Verify PDF ingestion followed by successful evidence passage retrieval."""
        # 1. Ingest
        ingest_res = self.agent.ingest_pdf(
            pdf_path=self.pdf_path,
            paper_id=55,
            filename="sample.pdf",
            doi="10.1234/medrag",
            allowed_dir=self.temp_dir
        )
        self.assertEqual(ingest_res["status"], "success")
        self.assertGreater(ingest_res["chunks_indexed"], 0)

        # 2. Retrieve
        retrieve_res = self.agent.retrieve_passages("medical benchmarks", top_k=3)
        self.assertIn(retrieve_res["status"], ("success", "low_confidence"))
        self.assertGreater(retrieve_res["count"], 0)

        # Validate passage structure
        p0 = retrieve_res["passages"][0]
        self.assertEqual(p0["paper_id"], 55)
        self.assertEqual(p0["filename"], "sample.pdf")
        self.assertEqual(p0["page_number"], 1)
        self.assertEqual(p0["doi"], "10.1234/medrag")
        self.assertIn("Evaluation benchmarks", p0["passage_text"])
        self.assertIn("relevance_score", p0)

    def test_duplicate_pdf_ingestion_prevented(self):
        """Verify re-ingesting the same PDF does not duplicate stored chunks."""
        res1 = self.agent.ingest_pdf(self.pdf_path, paper_id=99, allowed_dir=self.temp_dir)
        self.assertEqual(res1["status"], "success")
        initial_chunks = self.store.index.ntotal

        res2 = self.agent.ingest_pdf(self.pdf_path, paper_id=99, allowed_dir=self.temp_dir)
        self.assertEqual(res2["status"], "already_indexed")
        self.assertEqual(res2["chunks_indexed"], 0)
        self.assertEqual(self.store.index.ntotal, initial_chunks)


if __name__ == "__main__":
    unittest.main()
