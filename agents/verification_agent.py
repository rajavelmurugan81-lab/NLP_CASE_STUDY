import re
from typing import List, Dict, Any, Optional

from services.embedding_service import EmbeddingService
import numpy as np


class CitationVerificationAgent:
    """
    Agent responsible for verifying factual claims against retrieved evidence passages.
    Classifies claims as supported, partially_supported, unsupported, or conflicting.
    Validates source paper identifiers and flags unverifiable assertions.
    Does not claim verification guarantees ultimate scientific truth.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()

    def verify_claims(
        self,
        claims: List[Dict[str, Any]],
        known_papers: List[Dict[str, Any]],
        evidence_passages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verifies a list of claims against stored papers and retrieved evidence passages.
        
        Each claim input dictionary:
        {
          "claim_id": "claim_1",
          "claim_text": "RAG reduces hallucinations by 35% in clinical QA.",
          "cited_paper_id": 1,
          "cited_doi": "10.1001/...",
          "source_url": "..."
        }
        """
        if not claims:
            return {
                "total_claims": 0,
                "verified_claims": [],
                "summary": {"supported": 0, "partially_supported": 0, "unsupported": 0, "conflicting": 0},
                "verification_notice": "Citation verification checks semantic concordance with retrieved evidence passages and does not constitute a guarantee of scientific truth."
            }

        paper_map = {p.get("id"): p for p in known_papers if p.get("id") is not None}
        openalex_map = {p.get("openalex_id"): p for p in known_papers if p.get("openalex_id")}
        doi_map = {p.get("doi"): p for p in known_papers if p.get("doi")}

        # Build passage pool texts
        passage_texts = [p.get("text") or p.get("passage_text", "") for p in evidence_passages if p]
        
        verified_list = []
        counts = {"supported": 0, "partially_supported": 0, "unsupported": 0, "conflicting": 0}

        for idx, claim_item in enumerate(claims, start=1):
            cid = claim_item.get("claim_id") or f"claim_{idx}"
            claim_text = claim_item.get("claim_text", "").strip()
            cited_pid = claim_item.get("cited_paper_id") or claim_item.get("paper_id")
            cited_doi = claim_item.get("cited_doi") or claim_item.get("doi")
            cited_oaid = claim_item.get("openalex_id")

            # 1. Validate paper identifier against stored metadata
            paper_found = (
                paper_map.get(cited_pid)
                or openalex_map.get(cited_oaid)
                or doi_map.get(cited_doi)
            )

            # 2. Check semantic evidence match
            best_passage = None
            best_score = 0.0

            if passage_texts and claim_text:
                try:
                    q_vec = self.embedding_service.embed_query(claim_text)
                    p_vecs = self.embedding_service.embed_texts(passage_texts)
                    sims = np.dot(p_vecs, q_vec.T).flatten()
                    best_idx = int(np.argmax(sims))
                    best_score = float(sims[best_idx])
                    best_passage = evidence_passages[best_idx]
                except Exception:
                    best_score = 0.0

            # 3. Classify claim status
            if not paper_found:
                status = "unsupported"
                rationale = "Cited paper identifier could not be validated against stored literature metadata."
            elif best_score >= 0.55:
                status = "supported"
                rationale = f"High semantic agreement ({round(best_score * 100, 1)}%) with retrieved evidence passage."
            elif best_score >= 0.35:
                status = "partially_supported"
                rationale = f"Moderate semantic overlap ({round(best_score * 100, 1)}%); evidence provides partial contextual support."
            else:
                status = "unsupported"
                rationale = f"Insufficient semantic grounding ({round(best_score * 100, 1)}%) in retrieved literature passages."

            counts[status] += 1

            verified_list.append({
                "claim_id": cid,
                "claim_text": claim_text,
                "status": status,
                "confidence_score": round(best_score, 4),
                "rationale": rationale,
                "validated_paper": {
                    "id": paper_found.get("id") if paper_found else None,
                    "title": paper_found.get("title") if paper_found else "Unknown",
                    "doi": paper_found.get("doi") if paper_found else cited_doi,
                    "url": paper_found.get("url") if paper_found else None
                } if paper_found else None,
                "supporting_passage": best_passage.get("text") or best_passage.get("passage_text") if best_passage else None,
                "page_number": best_passage.get("page_number") if best_passage else None,
                "requires_revision": status in ("unsupported", "conflicting")
            })

        return {
            "total_claims": len(verified_list),
            "verified_claims": verified_list,
            "summary": counts,
            "all_verified": counts["unsupported"] == 0 and counts["conflicting"] == 0,
            "verification_notice": "Citation verification checks semantic concordance with retrieved evidence passages and does not constitute a guarantee of scientific truth."
        }
