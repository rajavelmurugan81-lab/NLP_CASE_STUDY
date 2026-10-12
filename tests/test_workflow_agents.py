import os
import unittest
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np

from agents.planning_agent import PlanningAgent
from agents.evidence_extraction_agent import EvidenceExtractionAgent
from services.clustering_service import ClusteringService, cluster_papers
from agents.comparative_agent import ComparativeAgent
from agents.gap_analysis_agent import ResearchGapAgent
from agents.verification_agent import CitationVerificationAgent
from services.report_generator import ReportGenerator, LiteratureReviewGenerator
from agents.workflow_coordinator import WorkflowCoordinator
from app import app


class WorkflowAgentsTestCase(unittest.TestCase):

    def setUp(self):
        self.sample_papers = [
            {
                "id": 1,
                "title": "Dense Passage Retrieval for Clinical Question Answering",
                "authors": "Smith, J., Doe, A.",
                "published_year": 2023,
                "abstract": "We evaluate dense passage retrieval models on clinical benchmarks. Method: Dual-encoder BERT. Dataset: PubMedQA. Metrics: F1 score 84.2%. Limitation: High GPU memory requirements and poor performance on rare terms.",
                "doi": "10.1000/1",
                "openalex_id": "https://openalex.org/W1",
                "url": "https://doi.org/10.1000/1"
            },
            {
                "id": 2,
                "title": "Iterative Retrieval-Augmented Generation for Evidence Synthesis",
                "authors": "Johnson, K., Lee, M.",
                "published_year": 2024,
                "abstract": "Iterative retrieval refines queries to synthesize multi-document literature. Method: Multi-agent prompting with FAISS. Dataset: BioASQ. Metrics: BLEU 38.5. Limitation: Sensitivity to noisy retrieved distractors.",
                "doi": "10.1000/2",
                "openalex_id": "https://openalex.org/W2",
                "url": "https://doi.org/10.1000/2"
            }
        ]

    def test_planning_agent_decomposition(self):
        """Verify PlanningAgent generates structured subtopics, queries, and criteria."""
        planner = PlanningAgent()
        plan = planner.plan_research("How does hybrid RAG improve factual accuracy?", max_subtopics=3)
        self.assertIn("subtopics", plan)
        self.assertGreaterEqual(len(plan["subtopics"]), 1)
        self.assertLessEqual(len(plan["subtopics"]), 3)
        for st in plan["subtopics"]:
            self.assertTrue("subtopic" in st or "title" in st)
            self.assertIn("search_query", st)
            self.assertTrue("criteria" in st or "required_evidence" in st)

    def test_evidence_extraction_agent(self):
        """Verify EvidenceExtractionAgent extracts 8 academic dimensions."""
        extractor = EvidenceExtractionAgent()
        paper = self.sample_papers[0]
        evidence = extractor.extract_evidence(paper, save_to_db=False)
        self.assertEqual(evidence["paper_id"], 1)
        self.assertIn("objective", evidence)
        self.assertIn("methodology", evidence)
        self.assertIn("dataset", evidence)
        self.assertIn("limitations", evidence)
        self.assertNotEqual(evidence["limitations"], "Not reported in available text")

    def test_clustering_service(self):
        """Verify ClusteringService clusters papers and derives grounded descriptive labels."""
        mock_embed = MagicMock()
        mock_embed.embed_texts.return_value = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        clusterer = ClusteringService(embedding_service=mock_embed, num_clusters=2)
        clusters = clusterer.cluster_papers(self.sample_papers)
        self.assertGreaterEqual(len(clusters), 1)
        for c in clusters:
            self.assertIn("cluster_id", c)
            self.assertIn("label", c)
            self.assertIn("papers", c)

    def test_comparative_agent_matrix_and_markdown(self):
        """Verify ComparativeAgent builds structured comparison matrix and Markdown table."""
        agent = ComparativeAgent()
        extractor = EvidenceExtractionAgent()
        evidence_list = [extractor.extract_evidence(p, save_to_db=False) for p in self.sample_papers]

        matrix_rows = agent.build_comparison_matrix(evidence_list)
        self.assertEqual(len(matrix_rows), 2)
        self.assertIn("objective", matrix_rows[0])
        self.assertIn("limitations", matrix_rows[0])

        md_table = agent.generate_markdown_table(matrix_rows)
        self.assertIn("| Paper & Year |", md_table)
        self.assertIn("Dense Passage Retrieval", md_table)

    def test_research_gap_agent(self):
        """Verify ResearchGapAgent extracts grounded limitations and categorizes gaps."""
        gap_agent = ResearchGapAgent()
        extractor = EvidenceExtractionAgent()
        evidence_list = [extractor.extract_evidence(p, save_to_db=False) for p in self.sample_papers]

        gaps_res = gap_agent.identify_gaps(evidence_list)
        self.assertIn("research_gaps", gaps_res)
        gaps = gaps_res["research_gaps"]
        self.assertGreater(len(gaps), 0)
        for g in gaps:
            self.assertIn("title", g)
            self.assertIn("description", g)
            self.assertIn("supporting_evidence", g)
            self.assertIn("recommended_direction", g)

    def test_citation_verification_agent(self):
        """Verify CitationVerificationAgent classifies claims against evidence passages."""
        mock_embed = MagicMock()
        mock_embed.embed_texts.side_effect = lambda texts: np.ones((len(texts), 4), dtype=np.float32) / 2.0
        mock_embed.embed_query.return_value = np.ones((1, 4), dtype=np.float32) / 2.0

        verifier = CitationVerificationAgent(embedding_service=mock_embed)
        claims = [
            {
                "claim_id": "C-1",
                "claim_text": "Dense retrieval improves clinical accuracy on PubMedQA.",
                "cited_paper_id": 1
            }
        ]
        passages = [
            {
                "passage_text": "We evaluate dense passage retrieval models on clinical benchmarks such as PubMedQA.",
                "filename": "paper1.pdf",
                "page_number": 1
            }
        ]

        result = verifier.verify_claims(claims, self.sample_papers, passages)
        self.assertIn("verified_claims", result)
        verified = result["verified_claims"]
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0]["claim_id"], "C-1")
        self.assertEqual(verified[0]["status"], "supported")

    def test_report_generator(self):
        """Verify ReportGenerator generates structured literature review with real citations."""
        gen = ReportGenerator()
        matrix = [
            {
                "title": "Dense Retrieval for Clinical QA",
                "year": 2023,
                "objective": "Improve QA accuracy",
                "methodology": "Dual-encoder BERT",
                "dataset": "PubMedQA",
                "model": "BERT",
                "metrics": "F1 84.2%",
                "findings": "Significant improvements over BM25",
                "limitations": "High GPU footprint"
            }
        ]
        gaps = [
            {
                "title": "High compute cost in low-resource settings",
                "rationale": "Reported by Smith et al. (2023)",
                "future_direction": "Explore quantization"
            }
        ]
        claims = [
            {
                "claim_id": "C-1",
                "claim_text": "Dense retrieval outperforms BM25 on clinical text.",
                "status": "supported",
                "source_paper_title": "Dense Retrieval for Clinical QA",
                "source_page": 1,
                "evidence_passage": "Significant improvements over BM25"
            }
        ]

        report_md = gen.generate_markdown_review(
            research_question="How does dense retrieval impact clinical QA?",
            papers=self.sample_papers,
            comparison_matrix=matrix,
            thematic_clusters=[{"cluster_id": 0, "label": "Clinical QA", "papers": self.sample_papers}],
            research_gaps=gaps,
            verified_claims=claims
        )

        self.assertIn("Automated Literature Review:", report_md)
        self.assertIn("## 1. Introduction & Research Question", report_md)
        self.assertIn("## 5. Methodological & Empirical Comparison", report_md)
        self.assertIn("## 8. Potential Research Gaps", report_md)
        self.assertIn("## 9. Citation & Evidence Verification Audit", report_md)
        self.assertIn("## 10. References & Source Links", report_md)

    def test_workflow_coordinator_bounded_execution(self):
        """Verify WorkflowCoordinator runs pipeline bounded by max_retries without uncontrolled loop."""
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve_papers.return_value = self.sample_papers
        mock_retrieval.retrieve_by_citations.return_value = []

        mock_rag = MagicMock()
        mock_rag.embedding_service = MagicMock()
        mock_rag.embedding_service.embed_texts.side_effect = lambda texts: np.ones((len(texts), 4), dtype=np.float32)
        mock_rag.embedding_service.embed_query.return_value = np.ones((1, 4), dtype=np.float32)
        mock_rag.vector_store = MagicMock()
        mock_rag.ingest_pdf.return_value = {"status": "success", "chunks_indexed": 3}
        mock_rag.retrieve_passages.return_value = {"status": "success", "passages": []}

        mock_hybrid = MagicMock()
        mock_hybrid.retrieve_hybrid.return_value = []

        coordinator = WorkflowCoordinator(
            retrieval_agent=mock_retrieval,
            rag_agent=mock_rag,
            hybrid_retriever=mock_hybrid,
            max_retries=2
        )

        with patch("agents.workflow_coordinator.create_workflow_run", return_value=101), \
             patch("agents.workflow_coordinator.update_workflow_run"), \
             patch("agents.workflow_coordinator.save_literature_report"):

            result = coordinator.run_full_review("Sample question on clinical NLP")
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["papers_processed"], 2)
            self.assertIn("report_markdown", result)
            self.assertLessEqual(result["retries_performed"], 2)

    def test_llm_status_api(self):
        """Verify GET /api/llm/status returns 200 and provider status."""
        client = app.test_client()
        response = client.get("/api/llm/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("configured", data)
        self.assertIn("provider", data)

    def test_workflow_run_api_validation(self):
        """Verify POST /api/workflow/run returns 400 for empty question."""
        client = app.test_client()
        response = client.post("/api/workflow/run", json={"question": ""})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
