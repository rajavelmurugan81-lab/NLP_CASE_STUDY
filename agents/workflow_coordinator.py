from typing import Dict, Any, Optional, List

from agents.planning_agent import PlanningAgent
from agents.retrieval_agent import RetrievalAgent
from agents.rag_agent import RagAgent
from agents.evidence_extraction_agent import EvidenceExtractionAgent
from agents.comparative_agent import ComparativeAgent
from agents.gap_analysis_agent import ResearchGapAgent
from agents.verification_agent import CitationVerificationAgent
from services.hybrid_retriever import HybridRetriever
from services.clustering_service import cluster_papers
from services.report_generator import LiteratureReviewGenerator
from database.db import create_workflow_run, update_workflow_run, save_literature_report


class WorkflowCoordinator:
    """
    Coordinates the full 11-stage Agentic Literature Review pipeline:
    Question -> Planning -> Multi-Source Retrieval -> Hybrid Fusion -> Re-ranking ->
    RAG Evidence -> Extraction -> Clustering -> Comparison -> Gaps -> Verification -> Sufficiency Loop -> Report.
    """

    def __init__(
        self,
        planning_agent: Optional[PlanningAgent] = None,
        retrieval_agent: Optional[RetrievalAgent] = None,
        rag_agent: Optional[RagAgent] = None,
        extraction_agent: Optional[EvidenceExtractionAgent] = None,
        comparative_agent: Optional[ComparativeAgent] = None,
        gap_agent: Optional[ResearchGapAgent] = None,
        verification_agent: Optional[CitationVerificationAgent] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        report_generator: Optional[LiteratureReviewGenerator] = None,
        max_retries: int = 2
    ):
        self.planning_agent = planning_agent or PlanningAgent(max_retries=max_retries)
        self.retrieval_agent = retrieval_agent or RetrievalAgent(default_limit=5)
        self.rag_agent = rag_agent or RagAgent()
        self.extraction_agent = extraction_agent or EvidenceExtractionAgent()
        self.comparative_agent = comparative_agent or ComparativeAgent()
        self.gap_agent = gap_agent or ResearchGapAgent()
        self.verification_agent = verification_agent or CitationVerificationAgent(
            embedding_service=self.rag_agent.embedding_service
        )
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            vector_store=self.rag_agent.vector_store,
            embedding_service=self.rag_agent.embedding_service
        )
        self.report_generator = report_generator or LiteratureReviewGenerator()
        self.max_retries = max_retries

    def run_full_review(self, research_question: str) -> Dict[str, Any]:
        """
        Executes the iterative bounded literature review workflow.
        """
        # Step 1: Initialize Workflow Run in SQLite
        workflow_id = create_workflow_run(research_question)

        # Step 2: Planning
        update_workflow_run(workflow_id, "planning")
        plan = self.planning_agent.plan_research(research_question)
        update_workflow_run(workflow_id, "retrieving", results_data={"plan": plan})

        # Step 3: Multi-Source Retrieval with Bounded Iteration Loop
        accumulated_papers: List[Dict[str, Any]] = []
        known_ids = set()
        retry_count = 0
        evidence_sufficiency = "pending"

        subtopics = plan.get("subtopics", [])

        while retry_count <= self.max_retries:
            for sub in subtopics:
                query = sub.get("search_query", research_question)
                try:
                    res = self.retrieval_agent.retrieve_papers(query=query, limit=5, save_to_db=True)
                    papers_list = res.get("papers", []) if isinstance(res, dict) else (res if isinstance(res, list) else [])
                    for p in papers_list:
                        pid = p.get("openalex_id") or p.get("id")
                        if pid and pid not in known_ids:
                            known_ids.add(pid)
                            accumulated_papers.append(p)
                except Exception:
                    continue

            # Also discover papers via citations/references if available
            if accumulated_papers:
                try:
                    ref_papers = self.retrieval_agent.retrieve_by_citations(accumulated_papers[:3], max_references=3)
                    for rp in ref_papers:
                        rpid = rp.get("openalex_id") or rp.get("id")
                        if rpid and rpid not in known_ids:
                            known_ids.add(rpid)
                            accumulated_papers.append(rp)
                except Exception:
                    pass

            # Sufficiency evaluation
            if len(accumulated_papers) >= 3 or retry_count >= self.max_retries:
                if len(accumulated_papers) >= 3:
                    evidence_sufficiency = "sufficient"
                elif len(accumulated_papers) > 0:
                    evidence_sufficiency = "partially_sufficient"
                else:
                    evidence_sufficiency = "insufficient"
                break

            # If insufficient, generate refined queries and retry within bounds
            retry_count += 1
            update_workflow_run(workflow_id, f"retrying_attempt_{retry_count}", retry_count=retry_count)
            subtopics = [
                {"search_query": f"{research_question} review survey analysis", "title": "Refined Survey Retrieval"}
            ]

        # Step 4: Hybrid Passage Retrieval over indexed chunks
        hybrid_passages = []
        try:
            hybrid_passages = self.hybrid_retriever.retrieve_hybrid(
                query=research_question,
                top_k=6,
                rerank=True
            )
        except Exception:
            pass

        # Step 5: Structured Evidence Extraction
        update_workflow_run(workflow_id, "extracting_evidence")
        extracted_evidence_list = []
        for paper in accumulated_papers:
            # Match passages belonging to this paper if any
            paper_chunks = [p for p in hybrid_passages if p.get("paper_id") == paper.get("id")]
            ev = self.extraction_agent.extract_evidence(paper, full_text_chunks=paper_chunks)
            extracted_evidence_list.append(ev)

        # Step 6: Thematic Clustering
        update_workflow_run(workflow_id, "clustering")
        clusters_data = cluster_papers(
            papers=accumulated_papers,
            embedding_service=self.rag_agent.embedding_service,
            num_clusters=min(3, max(1, len(accumulated_papers)))
        )

        # Step 7: Comparative Reasoning
        update_workflow_run(workflow_id, "comparing")
        comparison_data = self.comparative_agent.compare_papers(extracted_evidence_list)

        # Step 8: Research Gap Analysis
        update_workflow_run(workflow_id, "gap_analysis")
        gaps_data = self.gap_agent.identify_gaps(extracted_evidence_list)

        # Step 9: Citation Verification Audit
        update_workflow_run(workflow_id, "verifying_citations")
        claims_to_audit = []
        for idx, ev in enumerate(extracted_evidence_list, start=1):
            if ev.get("findings") and ev.get("findings") != "Not reported in available text":
                claims_to_audit.append({
                    "claim_id": f"claim_{idx}",
                    "claim_text": f"{ev.get('title')}: {ev.get('findings')}",
                    "paper_id": ev.get("paper_id"),
                    "openalex_id": ev.get("openalex_id")
                })

        verification_data = self.verification_agent.verify_claims(
            claims=claims_to_audit,
            known_papers=accumulated_papers,
            evidence_passages=hybrid_passages
        )

        # Step 10: Final Literature Review Synthesis
        update_workflow_run(workflow_id, "generating_report")
        report_res = self.report_generator.generate_report(
            research_question=research_question,
            plan=plan,
            papers=accumulated_papers,
            clusters_data=clusters_data,
            comparison_data=comparison_data,
            gaps_data=gaps_data,
            verification_data=verification_data,
            evidence_sufficiency=evidence_sufficiency
        )

        # Step 11: Persist Report in SQLite
        save_literature_report(
            workflow_id=workflow_id,
            research_question=research_question,
            report_markdown=report_res["markdown"],
            report_json=report_res
        )

        final_workflow_status = "completed" if evidence_sufficiency != "insufficient" else "insufficient_evidence"
        final_results = {
            "workflow_id": workflow_id,
            "research_question": research_question,
            "status": "success" if evidence_sufficiency != "insufficient" else "insufficient_evidence",
            "workflow_status": final_workflow_status,
            "evidence_sufficiency": evidence_sufficiency,
            "retry_count": retry_count,
            "retries_performed": retry_count,
            "plan": plan,
            "papers_count": len(accumulated_papers),
            "papers_processed": len(accumulated_papers),
            "papers": accumulated_papers,
            "clusters": clusters_data,
            "thematic_clusters": clusters_data.get("clusters", []),
            "comparison": comparison_data,
            "comparison_matrix": comparison_data.get("comparison_matrix", []),
            "gaps": gaps_data,
            "research_gaps": gaps_data.get("research_gaps", []),
            "verification": verification_data,
            "verified_claims": verification_data.get("verified_claims", []),
            "report": report_res,
            "report_markdown": report_res.get("markdown", "")
        }

        update_workflow_run(workflow_id, final_workflow_status, results_data=final_results)
        return final_results
