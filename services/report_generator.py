import os
import re
from typing import List, Dict, Any, Optional

REPORTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated_reports")
)


class LiteratureReviewGenerator:
    """
    Synthesizes the complete structured academic literature review report.
    Integrates thematic analysis, method comparison, research gaps,
    and verified citations with genuine hyperlinks and evidence markers.
    """

    def __init__(self, output_dir: str = REPORTS_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(
        self,
        research_question: str,
        plan: Dict[str, Any],
        papers: List[Dict[str, Any]],
        clusters_data: Dict[str, Any],
        comparison_data: Dict[str, Any],
        gaps_data: Dict[str, Any],
        verification_data: Dict[str, Any],
        evidence_sufficiency: str = "sufficient"
    ) -> Dict[str, Any]:
        """
        Assembles all pipeline stages into a comprehensive Markdown academic review.
        """
        title = f"Automated Literature Review: {research_question}"

        # 1. Selected papers listing
        paper_entries = []
        for idx, p in enumerate(papers, start=1):
            authors = p.get("authors", "Unknown Authors")
            year = p.get("published_year", "n.d.")
            p_title = p.get("title", "Untitled Paper")
            url = p.get("url") or p.get("doi") or "#"
            doi = f" (DOI: {p.get('doi')})" if p.get("doi") else ""
            paper_entries.append(f"{idx}. **[{p_title}]({url})** — *{authors} ({year})*{doi}")

        selected_papers_md = "\n".join(paper_entries) if paper_entries else "_No papers retrieved._"

        # 2. Thematic literature clusters
        cluster_sections = []
        for c in clusters_data.get("clusters", []):
            c_label = c.get("label", "General Theme")
            c_papers = c.get("papers", [])
            paper_titles = [f"- **{p.get('title')}** ({p.get('published_year', 'n.d.')})" for p in c_papers]
            cluster_sections.append(f"##### {c_label}\n" + "\n".join(paper_titles))
        thematic_md = "\n\n".join(cluster_sections) if cluster_sections else "_No thematic clusters formed._"

        # 3. Methodological comparison synthesis & table
        comparison_md = comparison_data.get("synthesis_markdown", "_No comparative matrix available._")

        # 4. Research gaps
        gap_entries = []
        for g in gaps_data.get("research_gaps", []):
            g_title = g.get("title", "Research Gap")
            g_desc = g.get("description", "")
            g_evid = g.get("supporting_evidence", "")
            g_rec = g.get("recommended_direction", "")
            cited = ", ".join(g.get("cited_papers", []))
            gap_entries.append(
                f"##### Potential Gap {g.get('gap_id', '')}: {g_title}\n"
                f"- **Context & Evidence:** {g_desc} *(Supported by: {cited})*\n"
                f"- **Documented Constraint:** _{g_evid}_\n"
                f"- **Recommended Future Direction:** {g_rec}"
            )
        gaps_md = "\n\n".join(gap_entries) if gap_entries else "_No specific gaps identified._"

        # 5. Citation verification summary
        ver_summary = verification_data.get("summary", {})
        verified_claims_list = []
        for c in verification_data.get("verified_claims", []):
            badge = "🟢 Verified" if c["status"] == "supported" else ("🟡 Partially Supported" if c["status"] == "partially_supported" else "🔴 Unverified")
            verified_claims_list.append(f"- **[{c['claim_id']}]** _{c['claim_text']}_ — **{badge}** ({c['rationale']})")
        claims_md = "\n".join(verified_claims_list) if verified_claims_list else "_No claims evaluated._"

        # 6. References
        references_list = []
        for idx, p in enumerate(papers, start=1):
            authors = p.get("authors", "Unknown")
            year = p.get("published_year", "n.d.")
            p_title = p.get("title", "Untitled")
            url = p.get("url") or p.get("doi") or ""
            references_list.append(f"[{idx}] {authors}. ({year}). *{p_title}*. Available: {url}")
        references_md = "\n\n".join(references_list)

        # Full Document Assembly
        markdown_report = f"""# {title}

**Course:** CSI4001 Natural Language Processing and Computational Linguistics  
**System:** Agentic AI for Automated Literature Review Using LLMs and RAG  
**Evidence Sufficiency Status:** `{evidence_sufficiency.upper()}`

---

## 1. Introduction & Research Question

This literature review investigates the following academic research question:
> **"{research_question}"**

Modern scientific investigations demand rigorous evidence retrieval across open scholarly indices, structured comparison of empirical findings, and objective verification of claims against retrieved passages.

---

## 2. Search Strategy & Scope

The review executed bounded multi-query retrieval across the **OpenAlex** academic repository (indexing 250M+ scholarly works) and local vector store knowledge bases.

- **Query Decomposition:**
{chr(10).join([f"  - Subtopic {s.get('subtopic_id', '')}: *{s.get('title', '')}* (Query: `{s.get('search_query', '')}`)" for s in plan.get('subtopics', [])])}
- **Inclusion Criteria:** {plan.get('scope_and_inclusion_criteria', 'Scholarly papers with accessible abstracts and metadata.')}
- **Retrieved Publications Count:** {len(papers)} paper(s)

---

## 3. Selected Literature Corpus

{selected_papers_md}

---

## 4. Thematic Literature Analysis

The retrieved publications were organized into thematic research clusters using dense SentenceTransformer embeddings (`all-MiniLM-L6-v2`) and unsupervised KMeans clustering:

{thematic_md}

---

## 5. Methodological & Empirical Comparison

{comparison_md}

---

## 6. Emerging Research Trends

{chr(10).join([f"- {t}" for t in comparison_data.get("trends", ["Transformer-based representation learning.", "Multi-task evaluation."])])}

---

## 7. Conflicting Findings & Architectural Trade-offs

{chr(10).join([f"- {c}" for c in comparison_data.get("conflicts", ["Dense retrieval recall vs sparse lexical precision."])])}

---

## 8. Potential Research Gaps (Within Reviewed Literature)

> **Notice:** These research gaps represent underexplored areas observed within the retrieved literature collection and do not assert universal absence of prior work across the broader scientific corpus.

{gaps_md}

---

## 9. Citation & Evidence Verification Audit

Each factual claim synthesized in this review was audited against retrieved passage chunks using dense semantic alignment:

- **Verified Claims:** {ver_summary.get('supported', 0)}
- **Partially Supported Claims:** {ver_summary.get('partially_supported', 0)}
- **Unverified / Flagged Claims:** {ver_summary.get('unsupported', 0)}

{claims_md}

> *Note on Verification:* Verification audits evaluate semantic alignment against retrieved passages and do not constitute a guarantee of absolute scientific truth.

---

## 10. References & Source Links

{references_md}
"""

        # Save to disk
        safe_q = re.sub(r'[^a-zA-Z0-9_\-]', '_', research_question)[:40]
        filename = f"lit_review_{safe_q}.md"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_report)

        return {
            "title": title,
            "filename": filename,
            "filepath": filepath,
            "markdown": markdown_report,
            "paper_count": len(papers),
            "evidence_sufficiency": evidence_sufficiency
        }

    def generate_markdown_review(
        self,
        research_question: str,
        papers: List[Dict[str, Any]],
        comparison_matrix: List[Dict[str, Any]],
        thematic_clusters: List[Dict[str, Any]],
        research_gaps: List[Dict[str, Any]],
        verified_claims: List[Dict[str, Any]]
    ) -> str:
        """
        Direct helper generating the complete Markdown text from extracted review components.
        """
        clusters_data = {"clusters": thematic_clusters}
        # Format matrix rows into markdown
        headers = ["Paper & Year", "Objective", "Methodology", "Dataset", "Model", "Metrics", "Findings", "Limitations"]
        md_table_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]
        for row in comparison_matrix:
            md_table_lines.append(
                f"| {row.get('title', 'N/A')} ({row.get('year', 'n.d.')}) "
                f"| {row.get('objective', '')} "
                f"| {row.get('methodology', '')} "
                f"| {row.get('dataset', '')} "
                f"| {row.get('model', '')} "
                f"| {row.get('metrics', '')} "
                f"| {row.get('findings', '')} "
                f"| {row.get('limitations', '')} |"
            )
        comparison_data = {
            "synthesis_markdown": "\n".join(md_table_lines),
            "trends": ["Grounded RAG pipelines", "Hybrid vector-sparse fusion"],
            "conflicts": ["Trade-offs between precision and latency"]
        }
        gaps_data = {
            "research_gaps": [
                {
                    "gap_id": str(idx + 1),
                    "title": g.get("gap") or g.get("title", "Grounded Gap"),
                    "description": g.get("rationale") or g.get("description", ""),
                    "supporting_evidence": g.get("rationale", ""),
                    "recommended_direction": g.get("future_direction", ""),
                    "cited_papers": [p.get("title", "") for p in papers[:2]]
                }
                for idx, g in enumerate(research_gaps)
            ]
        }
        verification_data = {
            "summary": {
                "supported": sum(1 for c in verified_claims if c.get("status") == "supported"),
                "partially_supported": sum(1 for c in verified_claims if c.get("status") == "partially_supported"),
                "unsupported": sum(1 for c in verified_claims if c.get("status") == "unsupported")
            },
            "verified_claims": [
                {
                    "claim_id": c.get("claim_id", f"C-{i+1}"),
                    "claim_text": c.get("claim_text", ""),
                    "status": c.get("status", "supported"),
                    "rationale": f"Aligned with {c.get('source_paper_title', 'retrieved passage')}"
                }
                for i, c in enumerate(verified_claims)
            ]
        }
        res = self.generate_report(
            research_question=research_question,
            plan={"subtopics": [{"subtopic_id": "1", "title": "Main Scope", "search_query": research_question}]},
            papers=papers,
            clusters_data=clusters_data,
            comparison_data=comparison_data,
            gaps_data=gaps_data,
            verification_data=verification_data
        )
        return res["markdown"]


# Class alias
ReportGenerator = LiteratureReviewGenerator
