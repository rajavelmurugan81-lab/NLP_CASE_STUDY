"""
services package:
Contains business logic, external integrations, PDF extraction, chunking, embeddings,
FAISS vector storage, BM25 keyword search, hybrid retrieval, LLM integration, clustering, and report generation.
"""
from .openalex_service import (
    search_works,
    fetch_work_by_id,
    reconstruct_abstract,
    OpenAlexError,
    OpenAlexConnectionError,
    OpenAlexAPIError,
)
from .pdf_service import (
    extract_text_from_pdf,
    validate_and_download_pdf,
    safe_resolve_pdf_path,
    PDFProcessingError,
    PDFEncryptedError,
    PDFCorruptedError
)
from .chunking_service import (
    clean_text,
    split_text_into_chunks,
    chunk_document_pages
)
from .embedding_service import (
    EmbeddingService,
    EmbeddingError,
    ModelLoadError
)
from .vector_store import (
    FaissVectorStore,
    VectorStoreError,
    VectorStoreInconsistencyError
)
from .bm25_service import BM25Service
from .hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from .llm_service import LLMService
from .clustering_service import cluster_papers, ClusteringService
from .report_generator import LiteratureReviewGenerator, ReportGenerator

__all__ = [
    "search_works",
    "fetch_work_by_id",
    "reconstruct_abstract",
    "OpenAlexError",
    "OpenAlexConnectionError",
    "OpenAlexAPIError",
    "extract_text_from_pdf",
    "validate_and_download_pdf",
    "safe_resolve_pdf_path",
    "PDFProcessingError",
    "PDFEncryptedError",
    "PDFCorruptedError",
    "clean_text",
    "split_text_into_chunks",
    "chunk_document_pages",
    "EmbeddingService",
    "EmbeddingError",
    "ModelLoadError",
    "FaissVectorStore",
    "VectorStoreError",
    "VectorStoreInconsistencyError",
    "BM25Service",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "LLMService",
    "cluster_papers",
    "ClusteringService",
    "LiteratureReviewGenerator",
    "ReportGenerator",
]
