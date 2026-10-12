import os
from typing import List, Optional
import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Base exception for embedding service failures."""
    pass


class ModelLoadError(EmbeddingError):
    """Raised when the SentenceTransformer model fails to load."""
    pass


class EmbeddingService:
    """
    Embedding service utilizing SentenceTransformers.
    Generates normalized dense vector embeddings for indexing and querying.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
        self._model = None
        self._dimension = None

    def _get_model(self):
        """
        Lazily loads the SentenceTransformer model instance on first use.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load embedding model '{self.model_name}': {str(exc)}"
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        """
        Returns the embedding dimension for the configured model (e.g. 384 for all-MiniLM-L6-v2).
        """
        if self._dimension is not None:
            return self._dimension

        if self._model is not None:
            try:
                self._dimension = self._model.get_sentence_embedding_dimension()
                return self._dimension
            except Exception:
                pass

        # Fallback inspection by encoding a test token
        try:
            model = self._get_model()
            test_vec = model.encode(["test"], convert_to_numpy=True)
            self._dimension = test_vec.shape[1]
            return self._dimension
        except Exception:
            # Default dimension for all-MiniLM-L6-v2
            return 384

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of chunk texts into L2-normalized float32 numpy array.
        Normalizing embeddings ensures inner-product search in FAISS computes Cosine Similarity.
        """
        if not texts:
            dim = self.dimension
            return np.empty((0, dim), dtype=np.float32)

        # Sanitize texts
        clean_texts = [str(t) if t is not None else "" for t in texts]

        try:
            model = self._get_model()
            embeddings = model.encode(
                clean_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            # Ensure float32 for FAISS compatibility
            return embeddings.astype(np.float32)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Error generating embeddings for text batch: {str(exc)}") from exc

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encodes a single research query string into a 2D float32 normalized vector (shape: 1 x dim).
        """
        if not query or not str(query).strip():
            raise EmbeddingError("Query text cannot be empty.")

        query_clean = str(query).strip()
        try:
            model = self._get_model()
            query_vec = model.encode(
                [query_clean],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return query_vec.astype(np.float32)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Error generating embedding for query: {str(exc)}") from exc
