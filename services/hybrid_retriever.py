from typing import List, Dict, Any, Optional
import numpy as np

from services.bm25_service import BM25Service
from services.vector_store import FaissVectorStore
from services.embedding_service import EmbeddingService


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k_constant: int = 60,
    k: Optional[int] = None,
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Combines multiple ranked result lists using Reciprocal Rank Fusion (RRF).
    Formula: RRF_score(d) = sum_{m in Models} (1 / (k + rank_m(d)))
    Deduplicates items by chunk_id or text.
    """
    effective_k = k if k is not None else k_constant
    scores_by_id: Dict[str, float] = {}
    items_by_id: Dict[str, Dict[str, Any]] = {}
    ranks_by_id: Dict[str, Dict[str, int]] = {}

    for list_idx, rank_list in enumerate(ranked_lists):
        source_name = f"source_{list_idx}"
        for rank, item in enumerate(rank_list, start=1):
            chunk_id = item.get("chunk_id") or item.get("id") or str(hash(item.get("text", "")))
            
            if chunk_id not in items_by_id:
                items_by_id[chunk_id] = dict(item)
                scores_by_id[chunk_id] = 0.0
                ranks_by_id[chunk_id] = {}

            # Accumulate RRF reciprocal rank
            reciprocal_score = 1.0 / (effective_k + rank)
            scores_by_id[chunk_id] += reciprocal_score
            ranks_by_id[chunk_id][source_name] = rank

    # Sort chunks by RRF score descending
    sorted_chunk_ids = sorted(
        scores_by_id.keys(),
        key=lambda cid: scores_by_id[cid],
        reverse=True
    )

    fused_results = []
    for rank, cid in enumerate(sorted_chunk_ids, start=1):
        item = items_by_id[cid]
        item["rrf_score"] = round(scores_by_id[cid], 5)
        item["rrf_rank"] = rank
        item["ranks_by_source"] = ranks_by_id[cid]
        fused_results.append(item)

    if top_k is not None:
        return fused_results[:top_k]
    return fused_results


class HybridRetriever:
    """
    Hybrid Retrieval coordinator combining:
    1. FAISS dense semantic search (SentenceTransformers)
    2. BM25 sparse keyword search (rank-bm25)
    3. Citation network connections
    4. Reciprocal Rank Fusion (RRF k=60)
    5. Semantic Re-ranking
    """

    def __init__(
        self,
        vector_store: FaissVectorStore,
        embedding_service: EmbeddingService,
        bm25_service: Optional[BM25Service] = None,
        rrf_k: int = 60,
        k: Optional[int] = None
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.rrf_k = k if k is not None else rrf_k
        self.bm25_service = bm25_service or BM25Service(chunks=self.vector_store.chunks)

    def sync_bm25_with_vector_store(self) -> None:
        """Synchronizes BM25 index with currently stored chunks in FAISS."""
        self.bm25_service.index_chunks(self.vector_store.chunks)

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid dense + sparse retrieval, fuses with RRF, and applies semantic re-ranking.
        Preserves all source metadata, page numbers, and attribution.
        """
        if not query or not str(query).strip():
            return []

        # If vector store is empty, return empty list
        if self.vector_store.is_empty():
            return []

        # Ensure BM25 has latest chunks if count changed
        if len(self.bm25_service.chunks) != len(self.vector_store.chunks):
            self.sync_bm25_with_vector_store()

        candidate_k = max(top_k * 3, 15)

        # 1. Dense Semantic Search (FAISS)
        dense_results: List[Dict[str, Any]] = []
        try:
            query_vec = self.embedding_service.embed_query(query)
            dense_results = self.vector_store.search(query_vec, top_k=candidate_k)
            for rank, r in enumerate(dense_results, start=1):
                r["dense_rank"] = rank
                r["dense_score"] = r.get("relevance_score", 0.0)
        except Exception:
            dense_results = []

        # 2. Sparse Keyword Search (BM25)
        sparse_results: List[Dict[str, Any]] = []
        try:
            sparse_results = self.bm25_service.search(query, top_k=candidate_k)
            for rank, r in enumerate(sparse_results, start=1):
                r["sparse_rank"] = rank
        except Exception:
            sparse_results = []

        # If both fail, return empty
        if not dense_results and not sparse_results:
            return []

        # 3. Reciprocal Rank Fusion (RRF)
        ranked_lists = []
        if dense_results:
            ranked_lists.append(dense_results)
        if sparse_results:
            ranked_lists.append(sparse_results)

        fused_candidates = reciprocal_rank_fusion(ranked_lists, k_constant=self.rrf_k)

        # 4. Semantic Re-ranking
        if rerank and fused_candidates:
            fused_candidates = self._rerank_candidates(query, fused_candidates)

        for c in fused_candidates:
            if "passage_text" not in c:
                c["passage_text"] = c.get("text", "")
            if "retrieval_method" not in c:
                c["retrieval_method"] = "hybrid_rrf_reranked" if rerank else "hybrid_rrf"

        return fused_candidates[:top_k]

    def _rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic re-ranking of top candidates using dense embedding cosine similarities
        blended with RRF positional scores.
        """
        try:
            query_vec = self.embedding_service.embed_query(query)  # Shape: (1, dim)
            candidate_texts = [c.get("text", "") for c in candidates]
            cand_vecs = self.embedding_service.embed_texts(candidate_texts)  # Shape: (N, dim)

            # Cosine similarity (both are L2 normalized, so inner product = cosine)
            sim_scores = np.dot(cand_vecs, query_vec.T).flatten()

            for i, cand in enumerate(candidates):
                semantic_score = float(sim_scores[i])
                rrf_score = cand.get("rrf_score", 0.0)
                # Combined re-ranking score (0.7 semantic + 0.3 RRF)
                combined_score = round(0.7 * semantic_score + 0.3 * (rrf_score * 30), 4)
                cand["rerank_score"] = combined_score
                cand["relevance_score"] = round(semantic_score, 4)

            # Sort by re-rank score descending
            reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return reranked

        except Exception:
            # Fallback to pure RRF order if re-ranking encounters an issue
            return candidates
