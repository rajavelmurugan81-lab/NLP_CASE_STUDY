import os
from typing import List, Dict, Any, Optional

from services.openalex_service import (
    search_works,
    fetch_work_by_id,
    reconstruct_abstract,
    OpenAlexError,
    OpenAlexConnectionError,
    OpenAlexAPIError
)
from database.db import save_search_query, save_or_get_paper


class RetrievalAgent:
    """
    Agent responsible for querying academic repositories (OpenAlex),
    normalizing publication metadata, managing citation/reference graphs,
    and persisting deduplicated records in SQLite.
    """

    def __init__(self, default_limit: int = 10, openalex_service: Optional[Any] = None):
        self.default_limit = default_limit
        self.openalex_service = openalex_service

    def normalize_work(self, raw_work: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes a raw OpenAlex work object into a standardized schema.
        Handles missing fields safely, extracts citation metadata, and reconstructs abstracts.
        """
        # OpenAlex ID
        openalex_id = raw_work.get("id") or None

        # Title
        title = (
            raw_work.get("title")
            or raw_work.get("display_name")
            or "Untitled Document"
        ).strip()

        # Authors
        author_names = []
        for authorship in raw_work.get("authorships", []):
            if isinstance(authorship, dict):
                author_obj = authorship.get("author") or {}
                name = author_obj.get("display_name") or authorship.get("raw_author_name")
                if name:
                    author_names.append(name.strip())

        authors_str = ", ".join(author_names) if author_names else "Unknown Authors"

        # Publication Year
        pub_year = raw_work.get("publication_year")
        if pub_year is not None:
            try:
                pub_year = int(pub_year)
            except (ValueError, TypeError):
                pub_year = None

        # Reconstruct Abstract from inverted index
        abstract_inverted = raw_work.get("abstract_inverted_index")
        abstract = reconstruct_abstract(abstract_inverted)

        # DOI
        doi = raw_work.get("doi") or None
        if doi and isinstance(doi, str):
            doi = doi.strip()

        # Landing-page URL
        primary_loc = raw_work.get("primary_location") or {}
        landing_url = (
            primary_loc.get("landing_page_url")
            or doi
            or openalex_id
        )

        # Open Access PDF URL
        best_oa = raw_work.get("best_oa_location") or {}
        oa_info = raw_work.get("open_access") or {}
        
        pdf_url = best_oa.get("pdf_url") or primary_loc.get("pdf_url")
        if not pdf_url and oa_info.get("oa_url"):
            possible_oa_url = oa_info.get("oa_url")
            if possible_oa_url and possible_oa_url.lower().endswith(".pdf"):
                pdf_url = possible_oa_url

        is_oa = bool(oa_info.get("is_oa", False) or pdf_url)

        # Citations & References
        citation_count = raw_work.get("cited_by_count", 0)
        referenced_works = raw_work.get("referenced_works", [])
        if not isinstance(referenced_works, list):
            referenced_works = []

        return {
            "openalex_id": openalex_id,
            "doi": doi,
            "title": title,
            "authors": authors_str,
            "published_year": pub_year,
            "abstract": abstract,
            "url": landing_url,
            "pdf_url": pdf_url,
            "is_oa": is_oa,
            "citation_count": citation_count,
            "referenced_works": referenced_works
        }

    def retrieve_papers(
        self,
        query: str,
        limit: Optional[int] = None,
        save_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Coordinates the literature retrieval lifecycle:
        1. Queries OpenAlex API.
        2. Normalizes metadata.
        3. Saves search query and deduplicates papers in SQLite.
        """
        topic = query.strip()
        result_limit = limit if limit is not None else self.default_limit

        if not topic:
            return {
                "status": "error",
                "message": "Research query cannot be empty.",
                "papers": [],
                "count": 0
            }

        # 1. Fetch raw scholarly records
        raw_works = search_works(query=topic, limit=result_limit)

        # 2. Normalize metadata
        normalized_papers = [self.normalize_work(work) for work in raw_works]

        # 3. Save to database if requested
        query_id = None
        persisted_papers = []

        if save_to_db:
            status = "completed" if normalized_papers else "no_results"
            query_id = save_search_query(topic, status=status)

            for paper_data in normalized_papers:
                saved = save_or_get_paper(query_id, paper_data)
                saved["is_oa"] = paper_data.get("is_oa", False)
                persisted_papers.append(saved)
        else:
            persisted_papers = normalized_papers

        return {
            "status": "success",
            "query_id": query_id,
            "topic": topic,
            "count": len(persisted_papers),
            "papers": persisted_papers,
            "message": f"Successfully retrieved {len(persisted_papers)} paper(s) from OpenAlex."
            if persisted_papers else "No papers found matching the specified query."
        }

    def retrieve_by_citations(
        self,
        seed_papers: Any,
        max_references: int = 5,
        save_to_db: bool = True,
        max_citations: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves authentic referenced papers linked to seed papers using genuine OpenAlex IDs.
        Deduplicates against already known papers.
        """
        limit = max_citations if max_citations is not None else max_references
        if isinstance(seed_papers, dict):
            seed_papers = [seed_papers]
        elif not isinstance(seed_papers, list):
            return []

        discovered_papers = []
        known_ids = {p.get("openalex_id") for p in seed_papers if p.get("openalex_id")}
        fetch_fn = self.openalex_service.fetch_work_by_id if self.openalex_service and hasattr(self.openalex_service, "fetch_work_by_id") else fetch_work_by_id

        for paper in seed_papers:
            ref_ids = paper.get("referenced_works", [])
            for ref_id in ref_ids[:limit]:
                if ref_id in known_ids:
                    continue
                try:
                    raw_ref_work = fetch_fn(ref_id)
                    if raw_ref_work:
                        norm = self.normalize_work(raw_ref_work)
                        known_ids.add(norm.get("openalex_id"))
                        if save_to_db:
                            saved = save_or_get_paper(None, norm)
                            discovered_papers.append(saved)
                        else:
                            discovered_papers.append(norm)
                except Exception:
                    continue

        return discovered_papers
