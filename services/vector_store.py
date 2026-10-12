import os
import json
from typing import List, Dict, Any, Optional, Set
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

VECTOR_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_data")
)
DEFAULT_INDEX_PATH = os.path.join(VECTOR_DATA_DIR, "faiss_index.bin")
DEFAULT_METADATA_PATH = os.path.join(VECTOR_DATA_DIR, "chunk_metadata.json")


class VectorStoreError(Exception):
    """Base exception for vector store failures."""
    pass


class VectorStoreInconsistencyError(VectorStoreError):
    """Raised when FAISS index length does not match stored chunk metadata."""
    pass


class FaissVectorStore:
    """
    Persistent FAISS vector store with sidecar chunk metadata.
    Uses IndexFlatIP for cosine similarity on normalized embeddings.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        dimension: int = 384
    ):
        if faiss is None:
            raise VectorStoreError("The faiss library is required but not installed.")

        self.index_path = index_path or os.getenv("VECTOR_INDEX_PATH", DEFAULT_INDEX_PATH)
        self.metadata_path = metadata_path or os.getenv("VECTOR_METADATA_PATH", DEFAULT_METADATA_PATH)
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[Dict[str, Any]] = []
        self.ingested_doc_ids: Set[str] = set()

        # Attempt to load persistent state if present
        self.load()

    def is_empty(self) -> bool:
        """Returns True if the index contains 0 vectors."""
        return self.index.ntotal == 0

    def is_document_ingested(self, doc_identifier: str) -> bool:
        """Checks if a document identifier has already been indexed."""
        if not doc_identifier:
            return False
        return str(doc_identifier).strip() in self.ingested_doc_ids

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        doc_identifier: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Adds chunk vectors to FAISS and preserves chunk metadata.
        Deduplicates by doc_identifier to avoid repeated indexing of identical papers.
        """
        if not chunks:
            return {
                "status": "empty",
                "chunks_added": 0,
                "total_chunks": len(self.chunks),
                "message": "No chunks provided for ingestion."
            }

        # Check for duplicate document ingestion
        if doc_identifier and self.is_document_ingested(doc_identifier):
            return {
                "status": "duplicate_skipped",
                "chunks_added": 0,
                "total_chunks": len(self.chunks),
                "message": f"Document '{doc_identifier}' has already been indexed in the vector store."
            }

        if embeddings.shape[0] != len(chunks):
            raise VectorStoreError(
                f"Embedding count ({embeddings.shape[0]}) does not match chunk count ({len(chunks)})."
            )

        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Vector dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension})."
            )

        # Cast to float32
        norm_embeddings = embeddings.astype(np.float32)

        # Add to FAISS
        start_id = self.index.ntotal
        self.index.add(norm_embeddings)

        # Store metadata with assigned vector ID
        for idx, chunk in enumerate(chunks):
            record = dict(chunk)
            record["faiss_id"] = start_id + idx
            self.chunks.append(record)

        if doc_identifier:
            self.ingested_doc_ids.add(str(doc_identifier).strip())

        # Persist changes
        self.save()

        return {
            "status": "success",
            "chunks_added": len(chunks),
            "total_chunks": len(self.chunks),
            "doc_identifier": doc_identifier,
            "message": f"Successfully indexed {len(chunks)} chunk(s)."
        }

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most similar chunks for a given query vector.
        Returns empty list if the index is empty.
        """
        if self.is_empty():
            return []

        if query_vector.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Query vector dimension ({query_vector.shape[1]}) does not match index dimension ({self.dimension})."
            )

        k = max(1, min(int(top_k), self.index.ntotal))
        distances, indices = self.index.search(query_vector.astype(np.float32), k)

        matched_records = []
        for score, vector_idx in zip(distances[0], indices[0]):
            if 0 <= vector_idx < len(self.chunks):
                record = dict(self.chunks[vector_idx])
                record["relevance_score"] = round(float(score), 4)
                matched_records.append(record)

        return matched_records

    def save(self) -> None:
        """
        Persists the FAISS index to binary file and chunk metadata to JSON.
        """
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        # 1. Save FAISS binary index
        faiss.write_index(self.index, self.index_path)

        # 2. Save metadata JSON
        metadata_payload = {
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "ingested_doc_ids": list(self.ingested_doc_ids),
            "chunks": self.chunks
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2, ensure_ascii=False)

    def load(self) -> bool:
        """
        Loads existing FAISS index and chunk metadata.
        Validates consistency between index vector count and metadata records.
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            return False

        try:
            # 1. Load FAISS index
            loaded_index = faiss.read_index(self.index_path)

            # 2. Load chunk metadata
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata_payload = json.load(f)

            loaded_chunks = metadata_payload.get("chunks", [])
            loaded_doc_ids = set(metadata_payload.get("ingested_doc_ids", []))
            loaded_dimension = metadata_payload.get("dimension", self.dimension)

            # 3. Consistency check
            if loaded_index.ntotal != len(loaded_chunks):
                raise VectorStoreInconsistencyError(
                    f"Index/Metadata inconsistency: FAISS index contains {loaded_index.ntotal} vectors, "
                    f"but metadata file contains {len(loaded_chunks)} chunks."
                )

            self.index = loaded_index
            self.chunks = loaded_chunks
            self.ingested_doc_ids = loaded_doc_ids
            self.dimension = loaded_dimension
            return True

        except (json.JSONDecodeError, VectorStoreInconsistencyError):
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to load vector store: {str(exc)}") from exc

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about indexed documents and vectors."""
        return {
            "total_vectors": self.index.ntotal,
            "total_chunks": len(self.chunks),
            "ingested_documents_count": len(self.ingested_doc_ids),
            "ingested_documents": list(self.ingested_doc_ids),
            "dimension": self.dimension,
            "is_empty": self.is_empty()
        }
