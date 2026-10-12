import os
from typing import List, Dict, Any, Optional

from services.pdf_service import extract_text_from_pdf, PDFProcessingError
from services.chunking_service import chunk_document_pages, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from services.embedding_service import EmbeddingService, EmbeddingError
from services.vector_store import FaissVectorStore, VectorStoreError


class RagAgent:
    """
    RAG evidence retrieval agent for academic literature.
    Manages PDF knowledge base ingestion and semantic passage retrieval.
    Does not generate LLM synthesis (milestone restricted to evidence retrieval).
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[FaissVectorStore] = None
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or FaissVectorStore(
            dimension=self.embedding_service.dimension
        )

    def ingest_pdf(
        self,
        pdf_path: str,
        paper_id: Optional[int] = None,
        filename: Optional[str] = None,
        doi: Optional[str] = None,
        source_url: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        allowed_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        End-to-end ingestion pipeline for a research PDF:
        1. Extract text page-by-page.
        2. Chunk text with metadata preservation.
        3. Generate dense embeddings.
        4. Store vectors in FAISS with deduplication.
        """
        # Unique identifier for the document
        effective_filename = filename or os.path.basename(pdf_path)
        doc_identifier = f"paper_{paper_id}" if paper_id else effective_filename

        # Check if already indexed to prevent redundant processing
        if self.vector_store.is_document_ingested(doc_identifier):
            return {
                "status": "already_indexed",
                "doc_identifier": doc_identifier,
                "filename": effective_filename,
                "chunks_indexed": 0,
                "total_chunks": len(self.vector_store.chunks),
                "message": f"Document '{doc_identifier}' is already indexed in the vector store."
            }

        # 1. Page-by-page text extraction
        extraction = extract_text_from_pdf(
            pdf_path=pdf_path,
            paper_id=paper_id,
            filename=effective_filename,
            doi=doi,
            source_url=source_url,
            allowed_dir=allowed_dir
        )

        if not extraction.get("has_text", False):
            return {
                "status": extraction.get("status", "no_text"),
                "doc_identifier": doc_identifier,
                "filename": effective_filename,
                "total_pages": extraction.get("total_pages", 0),
                "chunks_indexed": 0,
                "message": extraction.get("message", "PDF contains no extractable text.")
            }

        # 2. Text chunking
        metadata = {
            "paper_id": paper_id,
            "filename": effective_filename,
            "doi": doi,
            "source_url": source_url
        }
        chunks = chunk_document_pages(
            pages=extraction["pages"],
            paper_metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not chunks:
            return {
                "status": "no_chunks",
                "doc_identifier": doc_identifier,
                "filename": effective_filename,
                "chunks_indexed": 0,
                "message": "Extracted text could not be divided into valid chunks."
            }

        # 3. Dense embeddings generation
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        # 4. Ingest into persistent vector store
        store_result = self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            doc_identifier=doc_identifier
        )

        return {
            "status": "success",
            "doc_identifier": doc_identifier,
            "filename": effective_filename,
            "total_pages": extraction.get("total_pages", 0),
            "chunks_indexed": len(chunks),
            "total_vectors_in_store": store_result.get("total_chunks", 0),
            "message": f"Successfully indexed {len(chunks)} chunk(s) from {extraction.get('total_pages', 0)} page(s)."
        }

    def retrieve_passages(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Retrieves top relevant evidence passages for a research question.
        Reports evidence sufficiency and empty knowledge base states.
        """
        query_clean = str(query).strip() if query else ""
        if not query_clean:
            return {
                "status": "error",
                "query": "",
                "count": 0,
                "passages": [],
                "message": "Query string cannot be empty."
            }

        # Check for empty vector store
        if self.vector_store.is_empty():
            return {
                "status": "empty_knowledge_base",
                "query": query_clean,
                "count": 0,
                "passages": [],
                "knowledge_base_stats": self.vector_store.get_stats(),
                "message": "The vector knowledge base is currently empty. Ingest research PDFs before performing semantic retrieval."
            }

        # Encode query
        query_vec = self.embedding_service.embed_query(query_clean)

        # Vector search
        raw_matches = self.vector_store.search(query_vec, top_k=top_k)

        # Format retrieved passages
        formatted_passages = []
        for match in raw_matches:
            score = match.get("relevance_score", 0.0)
            if score >= min_score:
                formatted_passages.append({
                    "chunk_id": match.get("chunk_id"),
                    "passage_text": match.get("text"),
                    "relevance_score": score,
                    "paper_id": match.get("paper_id"),
                    "filename": match.get("filename"),
                    "page_number": match.get("page_number"),
                    "doi": match.get("doi"),
                    "source_url": match.get("source_url")
                })

        # Evaluate evidence sufficiency
        if not formatted_passages:
            status = "insufficient_evidence"
            message = "No relevant literature passages were found matching your research question."
        elif formatted_passages[0]["relevance_score"] < 0.25:
            status = "low_confidence"
            message = (
                f"Retrieved {len(formatted_passages)} passage(s), but similarity confidence is low. "
                "Consider indexing more domain-specific papers on this topic."
            )
        else:
            status = "success"
            message = f"Successfully retrieved {len(formatted_passages)} relevant evidence passage(s)."

        return {
            "status": status,
            "query": query_clean,
            "count": len(formatted_passages),
            "passages": formatted_passages,
            "knowledge_base_stats": self.vector_store.get_stats(),
            "message": message
        }
