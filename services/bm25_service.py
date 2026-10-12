import re
from typing import List, Dict, Any, Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def tokenize_text(text: str) -> List[str]:
    """
    Lowercases and extracts alphanumeric words and scientific tokens for BM25 matching.
    """
    if not text or not isinstance(text, str):
        return []
    return re.findall(r'[a-zA-Z0-9_\-\.\+]+', text.lower())


class BM25Service:
    """
    BM25 Keyword Search Service over document chunks.
    Maintains an in-memory Okapi BM25 index over indexed chunk texts.
    """

    def __init__(self, chunks: Optional[List[Dict[str, Any]]] = None):
        self.chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25_index: Optional[Any] = None

        if chunks:
            self.index_chunks(chunks)

    def is_empty(self) -> bool:
        """Returns True if the BM25 index contains no documents."""
        return len(self.chunks) == 0 or self.bm25_index is None

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Indexes a list of chunk metadata records using BM25Okapi.
        """
        if not chunks:
            self.chunks = []
            self.tokenized_corpus = []
            self.bm25_index = None
            return

        self.chunks = list(chunks)
        self.tokenized_corpus = [
            tokenize_text(c.get("text", "")) for c in self.chunks
        ]

        if BM25Okapi and self.tokenized_corpus:
            # If all token lists are empty, add dummy token to avoid ZeroDivisionError
            safe_corpus = [tokens if tokens else ["<empty>"] for tokens in self.tokenized_corpus]
            self.bm25_index = BM25Okapi(safe_corpus)
        else:
            self.bm25_index = None

    # Alias for API compatibility
    build_index = index_chunks

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs BM25 keyword search for query against indexed chunks.
        Returns top_k matches ranked by BM25 score.
        """
        if self.is_empty() or not query or not str(query).strip():
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        if self.bm25_index is None:
            return []

        scores = self.bm25_index.get_scores(query_tokens)
        
        # Rank by score descending
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        k = max(1, min(int(top_k), len(self.chunks)))
        results = []

        for rank, idx in enumerate(scored_indices[:k], start=1):
            score = float(scores[idx])
            # Only include non-zero scoring chunks or top results if positive
            if score > 0.0 or rank == 1:
                item = dict(self.chunks[idx])
                item["bm25_score"] = round(score, 4)
                item["bm25_rank"] = rank
                results.append(item)

        return results
