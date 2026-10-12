import re
from typing import List, Dict, Any, Optional

from services.llm_service import LLMService


class PlanningAgent:
    """
    Planning Agent for automated literature review.
    Decomposes research questions into subtopics and targeted search queries,
    defines required evidence criteria, and establishes workflow bounds.
    """

    def __init__(self, llm_service: Optional[LLMService] = None, max_retries: int = 2):
        self.llm_service = llm_service or LLMService()
        self.max_retries = max_retries

    def plan_research(self, research_question: str, max_subtopics: Optional[int] = None) -> Dict[str, Any]:
        """
        Creates a structured literature review research plan.
        Generates subtopics, academic search queries, and evidence criteria.
        """
        question = str(research_question).strip()
        if not question:
            raise ValueError("Research question cannot be empty.")

        # Attempt LLM decomposition if configured
        plan = None
        if self.llm_service.is_configured():
            try:
                plan = self._plan_with_llm(question)
            except Exception:
                # Graceful fallback to heuristic planning
                pass

        if not plan:
            plan = self._plan_with_heuristics(question)

        # Standardize field keys and slice if requested
        if plan and "subtopics" in plan:
            for st in plan["subtopics"]:
                if "subtopic" not in st:
                    st["subtopic"] = st.get("title", "")
                if "criteria" not in st:
                    st["criteria"] = st.get("required_evidence", "")
            if max_subtopics is not None:
                plan["subtopics"] = plan["subtopics"][:max_subtopics]

        return plan

    def _plan_with_llm(self, question: str) -> Dict[str, Any]:
        """Uses LLM to decompose the research question into academic subtopics."""
        prompt = f"""
You are an expert NLP and computational linguistics literature review planner.
Decompose the following research question into 3-4 specific academic subtopics, generate targeted academic search queries, and define evidence requirements.

Research Question: "{question}"

Respond strictly with a JSON object in this format:
{{
  "research_question": "{question}",
  "subtopics": [
    {{
      "subtopic_id": 1,
      "title": "Subtopic title",
      "search_query": "Targeted OpenAlex search query",
      "required_evidence": "Specific evidence required (e.g. architecture details, benchmark results, dataset comparisons)"
    }}
  ],
  "scope_and_inclusion_criteria": "Peer-reviewed publications, arXiv preprints, empirical evaluations",
  "max_retries": {self.max_retries}
}}
"""
        return self.llm_service.generate_json(
            prompt=prompt,
            system_prompt="You are a rigorous academic literature review research planner."
        )

    def _plan_with_heuristics(self, question: str) -> Dict[str, Any]:
        """
        Rule-based NLP decomposition fallback when LLM is unavailable or offline.
        Deconstructs the question into fundamental literature aspects:
        1. Core Methodology and Architectures
        2. Empirical Evaluation and Benchmark Datasets
        3. Real-world Performance, Limitations and Vulnerabilities
        """
        clean_q = re.sub(r'[\?\.\!]', '', question).strip()
        
        subtopics = [
            {
                "subtopic_id": 1,
                "title": f"Methodological Foundations of {clean_q}",
                "search_query": f"{clean_q} architecture methodology",
                "required_evidence": "Algorithmic architectures, theoretical mechanisms, and model design."
            },
            {
                "subtopic_id": 2,
                "title": f"Empirical Benchmarks and Datasets for {clean_q}",
                "search_query": f"{clean_q} benchmark evaluation dataset",
                "required_evidence": "Experimental datasets, quantitative metrics (F1, BLEU, accuracy, latency), and baselines."
            },
            {
                "subtopic_id": 3,
                "title": f"Limitations, Hallucinations, and Challenges in {clean_q}",
                "search_query": f"{clean_q} limitations challenges error analysis",
                "required_evidence": "Documented failure modes, computational overhead, domain constraints, and open problems."
            }
        ]

        return {
            "research_question": question,
            "subtopics": subtopics,
            "scope_and_inclusion_criteria": "Scholarly publications indexed in OpenAlex, empirical studies, and verified abstracts.",
            "max_retries": self.max_retries,
            "mode": "heuristic"
        }
