import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from services.embedding_service import (
    EmbeddingService,
    EmbeddingError,
    ModelLoadError
)


class EmbeddingServiceTestCase(unittest.TestCase):

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_texts_normalization_and_shape(self, mock_model_cls):
        """Verify embeddings are L2-normalized float32 vectors with correct dimensions."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        # Return synthetic 2D embeddings
        raw_vecs = np.array([[1.0, 2.0, 3.0] * 128, [4.0, 5.0, 6.0] * 128], dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        norm_vecs = raw_vecs / norms
        mock_model.encode.return_value = norm_vecs
        mock_model_cls.return_value = mock_model

        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        self.assertEqual(service.dimension, 384)

        texts = ["Text chunk one", "Text chunk two"]
        embeddings = service.embed_texts(texts)

        self.assertEqual(embeddings.shape, (2, 384))
        self.assertEqual(embeddings.dtype, np.float32)
        # Check unit norm (cosine ready)
        self.assertAlmostEqual(float(np.linalg.norm(embeddings[0])), 1.0, places=4)

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_query_shape(self, mock_model_cls):
        """Verify single query vector has shape (1, dimension)."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)
        mock_model_cls.return_value = mock_model

        service = EmbeddingService()
        query_vec = service.embed_query("Clinical question answering")

        self.assertEqual(query_vec.shape, (1, 384))
        self.assertEqual(query_vec.dtype, np.float32)

    def test_embed_empty_query_raises_error(self):
        """Verify empty query string raises EmbeddingError."""
        service = EmbeddingService()
        with self.assertRaises(EmbeddingError):
            service.embed_query("")

    @patch("sentence_transformers.SentenceTransformer", side_effect=Exception("Failed to load weights"))
    def test_model_load_error_handling(self, mock_model_cls):
        """Verify model loading failures raise ModelLoadError."""
        service = EmbeddingService()
        with self.assertRaises(ModelLoadError):
            service.embed_texts(["Test"])


if __name__ == "__main__":
    unittest.main()
