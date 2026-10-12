import re
from typing import List, Dict, Any, Optional

from services.llm_service import LLMService
from database.db import save_extracted_evidence, get_extracted_evidence_by_paper_id


class EvidenceExtractionAgent:
    """
    Agent responsible for extracting structured academic findings from paper content:
    objective, methodology, dataset, architecture, metrics, findings, limitations, future work.
    Preserves verbatim supporting evidence passages and source identifiers.
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()

    def extract_evidence(
        self,
        paper: Dict[str, Any],
        full_text_chunks: Optional[List[Dict[str, Any]]] = None,
        save_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Extracts structured scientific dimensions from a paper's abstract or retrieved chunks.
        """
        paper_id = paper.get("id")
        openalex_id = paper.get("openalex_id")

        # Check if already extracted in SQLite
        if paper_id and save_to_db:
            existing = get_extracted_evidence_by_paper_id(paper_id)
            if existing:
                return existing

        # Compile available text
        text_sources = []
        if paper.get("abstract"):
            text_sources.append(f"ABSTRACT:\n{paper['abstract']}")

        if full_text_chunks:
            chunk_texts = [f"[Page {c.get('page_number', '?')}]: {c.get('text', '')}" for c in full_text_chunks[:8]]
            text_sources.append("EXTRACTED TEXT PASSAGES:\n" + "\n---\n".join(chunk_texts))

        combined_text = "\n\n".join(text_sources).strip()
        if not combined_text:
            return self._build_empty_evidence(paper)

        # Attempt LLM extraction if available
        if self.llm_service.is_configured():
            try:
                evidence = self._extract_with_llm(paper, combined_text)
                if save_to_db and paper_id:
                    save_extracted_evidence(evidence)
                return evidence
            except Exception:
                pass

        # Deterministic extraction based on sentence pattern matching
        evidence = self._extract_with_heuristics(paper, combined_text)
        if save_to_db and paper_id:
            save_extracted_evidence(evidence)
        return evidence

    def _extract_with_llm(self, paper: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Extracts structured fields using LLM with strict grounding."""
        prompt = f"""
You are an expert NLP scientific reviewer.
Extract structured academic information strictly from the provided text for paper: "{paper.get('title', 'Unknown')}".
Do not hallucinate or guess. If an aspect is not explicitly mentioned, output "Not reported in available text".

Paper Text:
\"\"\"
{text[:4000]}
\"\"\"

Respond strictly in JSON format with these exact keys:
{{
  "objective": "Concise statement of research objective",
  "methodology": "Proposed method or algorithmic approach",
  "dataset": "Datasets or corpora utilized, or 'Not reported in available text'",
  "architecture": "Model or neural network architecture, or 'Not reported in available text'",
  "metrics": "Evaluation metrics (e.g. Accuracy, F1, BLEU), or 'Not reported in available text'",
  "findings": "Primary empirical conclusions and quantitative gains",
  "limitations": "Documented limitations, failure cases, or constraints",
  "future_work": "Future directions mentioned by authors, or 'Not reported in available text'",
  "evidence_passages": ["Direct quote 1 from text", "Direct quote 2 from text"]
}}
"""
        result = self.llm_service.generate_json(prompt, system_prompt="You are a factual academic evidence extractor.")
        result["paper_id"] = paper.get("id")
        result["openalex_id"] = paper.get("openalex_id")
        result["title"] = paper.get("title")
        return result

    def _extract_with_heuristics(self, paper: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Regex and sentence-level evidence extraction from available text.
        Marks unknown fields explicitly as 'Not reported in available text'.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)

        def find_match(cues: List[str], default: str = "Not reported in available text") -> str:
            for s in sentences:
                s_lower = s.lower()
                for cue in cues:
                    if cue in s_lower and len(s.strip()) > 20:
                        return s.strip()
            return default

        objective = find_match(
            ["we propose", "in this work", "this paper presents", "aims to", "we introduce", "investigate", "we study"],
            default=f"Investigate {paper.get('title', 'the topic')}"
        )
        methodology = find_match(
            ["framework", "pipeline", "approach", "algorithm", "technique", "methodology", "using", "prompting"]
        )
        dataset = find_match(
            ["dataset", "corpus", "benchmark", "benchmarking", "evaluated on", "collected from"]
        )
        architecture = find_match(
            ["model", "transformer", "architecture", "llm", "neural network", "encoder", "decoder", "gpt", "bert"]
        )
        metrics = find_match(
            ["accuracy", "f1", "bleu", "rouge", "precision", "recall", "latency", "perplexity", "score"]
        )
        findings = find_match(
            ["results show", "improves", "demonstrates", "outperforms", "achieves", "significant", "findings indicate"]
        )
        limitations = find_match(
            ["limitation", "hallucination", "overhead", "vulnerability", "bias", "drawback", "however", "challenge"]
        )
        future_work = find_match(
            ["future work", "promising direction", "further research", "remains to be explored", "we plan"]
        )

        evidence_passages = [s for s in [objective, findings, limitations] if s != "Not reported in available text"]

        return {
            "paper_id": paper.get("id"),
            "openalex_id": paper.get("openalex_id"),
            "title": paper.get("title"),
            "objective": objective,
            "methodology": methodology,
            "dataset": dataset,
            "architecture": architecture,
            "metrics": metrics,
            "findings": findings,
            "limitations": limitations,
            "future_work": future_work,
            "evidence_passages": evidence_passages[:3]
        }

    def _build_empty_evidence(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "paper_id": paper.get("id"),
            "openalex_id": paper.get("openalex_id"),
            "title": paper.get("title"),
            "objective": f"Study of {paper.get('title', 'research topic')}",
            "methodology": "Not reported in available text",
            "dataset": "Not reported in available text",
            "architecture": "Not reported in available text",
            "metrics": "Not reported in available text",
            "findings": "Full text and abstract not accessible for this publication.",
            "limitations": "Not reported in available text",
            "future_work": "Not reported in available text",
            "evidence_passages": []
        }
