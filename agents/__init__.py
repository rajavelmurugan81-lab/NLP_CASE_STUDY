"""
agents package:
Autonomous multi-agent architecture for automated academic literature review:
- PlanningAgent: Decomposes research questions into subtopics and search criteria.
- RetrievalAgent: Gathers scholarly records from OpenAlex, deduplicating in SQLite.
- RagAgent: Ingests research PDFs and retrieves evidence passages from FAISS.
- EvidenceExtractionAgent: Extracts structured methodology, findings, and limitations.
- ComparativeAgent: Synthesizes cross-paper comparisons and matrices.
- ResearchGapAgent: Discovers grounded research gaps and future directions.
- CitationVerificationAgent: Audits claims against retrieved passages and source metadata.
- WorkflowCoordinator: Orchestrates bounded iterative literature review execution.
"""
from .retrieval_agent import RetrievalAgent
from .rag_agent import RagAgent
from .planning_agent import PlanningAgent
from .evidence_extraction_agent import EvidenceExtractionAgent
from .comparative_agent import ComparativeAgent
from .gap_analysis_agent import ResearchGapAgent
from .verification_agent import CitationVerificationAgent
from .workflow_coordinator import WorkflowCoordinator

__all__ = [
    "RetrievalAgent",
    "RagAgent",
    "PlanningAgent",
    "EvidenceExtractionAgent",
    "ComparativeAgent",
    "ResearchGapAgent",
    "CitationVerificationAgent",
    "WorkflowCoordinator"
]
