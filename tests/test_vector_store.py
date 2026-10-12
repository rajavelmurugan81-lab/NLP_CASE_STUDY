import os
import json
import unittest
import tempfile
import numpy as np

from services.vector_store import (
    FaissVectorStore,
    VectorStoreError,
    VectorStoreInconsistencyError
)


class VectorStoreTestCase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.temp_dir, "test_faiss.bin")
        self.metadata_path = os.path.join(self.temp_dir, "test_chunks.json")
        self.dimension = 4
        self.store = FaissVectorStore(
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            dimension=self.dimension
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

    def test_initial_empty_state(self):
        """Verify new store starts empty and searches return empty list without errors."""
        self.assertTrue(self.store.is_empty())
        dummy_query = np.ones((1, self.dimension), dtype=np.float32)
        matches = self.store.search(dummy_query, top_k=3)
        self.assertEqual(matches, [])

    def test_add_chunks_and_search_similarity(self):
        """Verify vector indexing and ranked retrieval with cosine similarity."""
        chunks = [
            {"chunk_id": "c1", "text": "Medical RAG benchmarks", "page_number": 1},
            {"chunk_id": "c2", "text": "Financial stock prediction", "page_number": 2}
        ]
        # Vectors: c1 aligned with [1, 0, 0, 0], c2 aligned with [0, 1, 0, 0]
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        result = self.store.add_chunks(chunks, vectors, doc_identifier="doc_101")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["chunks_added"], 2)
        self.assertFalse(self.store.is_empty())

        # Query vector close to c1: [0.99, 0.0, 0.0, 0.0]
        query_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        search_res = self.store.search(query_vec, top_k=1)

        self.assertEqual(len(search_res), 1)
        self.assertEqual(search_res[0]["chunk_id"], "c1")
        self.assertAlmostEqual(search_res[0]["relevance_score"], 1.0, places=2)

    def test_duplicate_document_ingestion_skipped(self):
        """Verify duplicate ingestion of same document identifier is prevented."""
        chunks = [{"chunk_id": "c1", "text": "Text"}]
        vectors = np.ones((1, self.dimension), dtype=np.float32)

        # First ingestion
        res1 = self.store.add_chunks(chunks, vectors, doc_identifier="paper_123")
        self.assertEqual(res1["status"], "success")

        # Second ingestion with identical identifier
        res2 = self.store.add_chunks(chunks, vectors, doc_identifier="paper_123")
        self.assertEqual(res2["status"], "duplicate_skipped")
        self.assertEqual(res2["chunks_added"], 0)
        self.assertEqual(len(self.store.chunks), 1)

    def test_persistence_save_and_reload(self):
        """Verify FAISS binary and chunk metadata survive reload with validated consistency."""
        chunks = [{"chunk_id": "c1", "text": "Persisted text"}]
        vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.store.add_chunks(chunks, vectors, doc_identifier="doc_test")

        # Create new instance pointing to same files
        reloaded_store = FaissVectorStore(
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            dimension=self.dimension
        )

        self.assertFalse(reloaded_store.is_empty())
        self.assertEqual(len(reloaded_store.chunks), 1)
        self.assertEqual(reloaded_store.chunks[0]["chunk_id"], "c1")
        self.assertTrue(reloaded_store.is_document_ingested("doc_test"))

    def test_inconsistent_index_raises_error(self):
        """Verify vector count mismatch between index and metadata triggers error."""
        # Save valid store
        chunks = [{"chunk_id": "c1", "text": "Text"}]
        vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.store.add_chunks(chunks, vectors, doc_identifier="doc_err")

        # Corrupt metadata JSON by deleting chunks while FAISS has 1 vector
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump({"dimension": 4, "total_vectors": 0, "ingested_doc_ids": [], "chunks": []}, f)

        with self.assertRaises(VectorStoreInconsistencyError):
            FaissVectorStore(
                index_path=self.index_path,
                metadata_path=self.metadata_path,
                dimension=self.dimension
            )


if __name__ == "__main__":
    unittest.main()
