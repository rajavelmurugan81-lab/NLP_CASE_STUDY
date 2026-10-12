# Agentic AI for Automated Literature Review Using LLMs and RAG

**Course:** CSI4001 Natural Language Processing and Computational Linguistics  
**Project Status:** Complete Multi-Agent System Architecture Operational (Milestones 1–4)

---

## 1. System Overview

Conducting rigorous academic literature reviews requires deconstructing research questions, retrieving scholarly publications, extracting structured methodology and findings, clustering emerging themes, synthesizing cross-paper comparison matrices, identifying evidence-grounded research gaps, verifying factual assertions, and compiling an academic review with inline citations.

This system implements an **Autonomous Multi-Agent AI Pipeline** combining **Dense Vector Search (FAISS)**, **Sparse Keyword Search (BM25 Okapi)**, **Reciprocal Rank Fusion (RRF with $k=60$)**, **Semantic Re-ranking**, **KMeans Thematic Clustering**, and **Modular Large Language Models (LLMs)** with deterministic offline fallback heuristics.

### Key Architecture Pillars:
1. **Planning Agent:** Deconstructs research questions into specialized subtopics and targeted search queries, establishes evidence criteria, and orchestrates an iterative bounded loop ($max\_retries=2$).
2. **Multi-Source Retrieval Agent:** Queries the OpenAlex academic API (indexing 250M+ scholarly works) and crawls genuine citation/reference graphs without fabricating papers or metadata.
3. **Hybrid RAG Knowledge Base:**
   - Extracts page-by-page text from academic PDFs via **PyMuPDF** (`fitz`).
   - Slices text into overlapping, boundary-aware scientific chunks with page-level attribution.
   - Generates L2-normalized dense embeddings with **SentenceTransformers** (`all-MiniLM-L6-v2`).
   - Indexes chunks in **FAISS** (`IndexFlatIP`) for cosine similarity.
   - Indexes chunks with **BM25Okapi** for sparse scientific terminology matching.
   - Fuses rankings using **Reciprocal Rank Fusion (RRF $k=60$)** and applies dense semantic re-ranking.
4. **Evidence Extraction Agent:** Extracts 8 structured dimensions for each publication:
   - Research Objective
   - Methodology
   - Dataset
   - Model / Architecture
   - Evaluation Metrics
   - Main Findings
   - Documented Limitations
   - Future Work
5. **Thematic Clustering Service:** Unsupervised dense KMeans clustering on paper embeddings paired with TF-IDF grounded descriptive theme labels.
6. **Comparative Reasoning Agent:** Constructs structured cross-paper comparison matrix tables and synthesizes methodology convergence and architectural trade-offs.
7. **Research-Gap Discovery Agent:** Discovers research gaps grounded strictly within the reviewed paper collection's documented limitations, suggesting actionable future directions without unfounded universal novelty assertions.
8. **Citation & Evidence Verification Agent:** Audits synthesized factual claims against retrieved text passages, classifying each claim as `supported`, `partially_supported`, `unsupported`, or `conflicting` with semantic support scores.
9. **Literature Review Generator:** Synthesizes structured academic literature reviews in Markdown with genuine inline citations, verified claim audits, and reference bibliographies.

---

## 2. Directory Structure

```text
NLP_CASE_PROJECT/
├── app.py                         # Flask web app and complete REST API endpoints
├── requirements.txt               # Dependencies (Flask, PyMuPDF, sentence-transformers, faiss-cpu, rank-bm25, scikit-learn, openai)
├── .env.example                   # Template for environment configuration
├── .env                           # Local environment config (PORT=5001, models, storage)
├── .gitignore                     # Ignores venv, uploads, vector_data, reports, and local DBs
├── README.md                      # Comprehensive project documentation and evaluation manual
│
├── agents/                        # Autonomous Multi-Agent Pipeline
│   ├── __init__.py                # Package exports
│   ├── planning_agent.py          # Question decomposition & bounded retry limits
│   ├── retrieval_agent.py         # OpenAlex query coordinator & citation-graph discovery
│   ├── rag_agent.py               # End-to-end PDF ingestion & dense passage retrieval
│   ├── evidence_extraction_agent.py # 8-dimension structured academic evidence extractor
│   ├── comparative_agent.py       # Comparative matrix table and empirical synthesis
│   ├── gap_analysis_agent.py      # Grounded research gap discovery & future directions
│   ├── verification_agent.py      # Semantic claim auditor & citation grounding classifier
│   └── workflow_coordinator.py    # Master 11-stage autonomous orchestrator
│
├── services/                      # Core Business Logic & NLP Engine
│   ├── __init__.py                # Package exports
│   ├── openalex_service.py        # OpenAlex REST client & inverted-index abstract reconstruction
│   ├── pdf_service.py             # PyMuPDF page-by-page extraction, path safety, and corruption handling
│   ├── chunking_service.py        # Scientific text cleaning and boundary-aware overlapping chunking
│   ├── embedding_service.py       # SentenceTransformers embedding generation (float32 normalized)
│   ├── vector_store.py            # Persistent local FAISS vector store with sidecar JSON metadata
│   ├── bm25_service.py            # BM25Okapi keyword search over document chunks
│   ├── hybrid_retriever.py        # Dense-sparse fusion via Reciprocal Rank Fusion (k=60) + Re-ranking
│   ├── llm_service.py             # Configurable LLM service (OpenAI API + heuristic fallback mode)
│   ├── clustering_service.py      # Dense KMeans paper clustering & TF-IDF label generation
│   └── report_generator.py        # Structured literature review Markdown generator & disk persistence
│
├── database/                      # SQLite Relational Persistence
│   ├── __init__.py
│   ├── db.py                      # Schema & queries (papers, search_queries, workflow_runs, extracted_evidence, literature_reports)
│   └── literature_review.db       # Persistent SQLite database
│
├── templates/                     # Jinja2 Templates
│   └── index.html                 # Academic Dashboard (Discovery, Hybrid RAG, Multi-Agent Review tabs)
│
├── static/                        # Dashboard Assets
│   ├── css/
│   │   └── style.css              # Academic theme, responsive layout, stepper pills, verification badges
│   └── js/
│       └── app.js                 # Async UI controller, health check, hybrid search, pipeline runner, report viewer/copy
│
├── uploads/                       # Storage for user-uploaded research PDFs
├── vector_data/                   # Persistent local FAISS binary store & chunk metadata
│   ├── faiss_index.bin
│   └── chunk_metadata.json
├── generated_reports/             # Persistent Markdown literature review reports
└── tests/                         # Offline Automated Test Suite (60 tests, 100% pass)
    ├── __init__.py
    ├── test_health.py             # Flask route and health check tests
    ├── test_openalex_service.py   # OpenAlex client and inverted-index tests
    ├── test_retrieval_agent.py    # Metadata normalization and deduplication tests
    ├── test_search_api.py         # Search endpoint integration tests
    ├── test_pdf_service.py        # PyMuPDF extraction, corruption, and path traversal tests
    ├── test_chunking_service.py   # Scientific cleaning, chunk overlap, and metadata attribution tests
    ├── test_embedding_service.py  # Embedding normalization, shapes, and load failure tests
    ├── test_vector_store.py       # FAISS persistence, reload, and consistency tests
    ├── test_rag_agent.py          # RAG ingestion and passage retrieval tests
    ├── test_rag_api.py            # Upload, ingest, and retrieve API tests
    ├── test_hybrid_retriever.py   # BM25, RRF (k=60), re-ranking, and citation discovery tests
    └── test_workflow_agents.py    # Planning, extraction, clustering, comparison, gaps, verification, workflow tests
```

---

## 3. Setup and Installation (Windows PowerShell / Command Prompt)

### Step 3.1: Navigate to Project Directory
```powershell
cd c:\Users\rajav\OneDrive\Desktop\NLP_CASE_PROJECT
```

### Step 3.2: Activate the Virtual Environment
In PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If script execution is restricted on Windows, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3.3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3.4: Environment Configuration
Ensure `.env` exists (copied from `.env.example`):
```env
PORT=5001
FLASK_DEBUG=False
OPENALEX_API_KEY=
OPENALEX_EMAIL=your_email@institution.edu
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_INDEX_PATH=vector_data/faiss_index.bin
VECTOR_METADATA_PATH=vector_data/chunk_metadata.json
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini
```
> **Note on LLM Execution:** If `OPENAI_API_KEY` is not provided, the entire multi-agent pipeline operates autonomously in deterministic **Offline Heuristic Mode**, using rule-based NLP extraction, TF-IDF clustering labels, and algorithmic synthesis without requiring external API credits.

---

## 4. Running the Automated Test Suite

All 60 tests execute offline with zero external internet access or model download dependencies:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Test Coverage Summary:**
- 60 passing tests across 12 test modules.
- 0 failures, 0 errors, 0 regressions.
- Execution time: ~8.8 seconds.

---

## 5. Running the Application

Start the Flask application:
```powershell
.\venv\Scripts\python.exe app.py
```

The application runs on `http://127.0.0.1:5001`:
```text
Starting Literature Review Flask server on http://127.0.0.1:5001
 * Running on http://127.0.0.1:5001 (Press CTRL+C to quit)
```

Open your browser and navigate to:
**[http://127.0.0.1:5001](http://127.0.0.1:5001)**

---

## 6. REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | System health check and status confirmation |
| `/api/llm/status` | `GET` | LLM provider readiness and active execution mode |
| `/api/rag/stats` | `GET` | FAISS vector count and ingested document statistics |
| `/api/search` | `POST` | Real academic paper retrieval via OpenAlex with deduplication |
| `/api/upload` | `POST` | Multipart upload for academic PDF documents |
| `/api/ingest` | `POST` | Ingests uploaded PDF: PyMuPDF extraction, chunking, embeddings, FAISS indexing |
| `/api/retrieve` | `POST` | Dense semantic passage retrieval (FAISS cosine similarity) |
| `/api/retrieve/hybrid`| `POST` | Hybrid retrieval fusing FAISS dense and BM25 sparse via RRF ($k=60$) + Re-ranking |
| `/api/plan` | `POST` | Decomposes research question into subtopics and search queries |
| `/api/cluster` | `POST` | Unsupervised dense KMeans clustering of retrieved literature |
| `/api/compare` | `POST` | Extracts 8-dimension evidence and generates comparative matrix |
| `/api/gaps` | `POST` | Identifies research gaps grounded in cited limitations |
| `/api/verify` | `POST` | Audits synthesized claims against evidence passages |
| `/api/workflow/run` | `POST` | Executes complete 11-stage autonomous agentic literature review |
| `/api/workflow/<id>` | `GET` | Retrieves intermediate stage state and artifacts for a workflow run |
| `/api/report/latest` | `GET` | Retrieves most recently generated literature review report |
| `/api/report/export` | `GET` | Downloads latest literature review report as a `.md` Markdown file |

---

## 7. Interactive Dashboard Guide

The web dashboard is organized into 3 tabs:

### Tab 1: Literature Discovery (OpenAlex)
- Search over 250M+ scholarly works by topic or keywords.
- View real titles, authors, years, DOIs, and reconstructed abstracts.
- Direct links to publisher landing pages and Open Access PDFs.
- Automatic SQLite persistence and deduplication.

### Tab 2: RAG Knowledge Base & Hybrid Retrieval
- **Upload & Ingest:** Upload academic PDFs to safely parse text via PyMuPDF, create overlapping scientific chunks, compute MiniLM embeddings, and store in FAISS.
- **Retrieval Method Selector:** Choose between:
  - `Hybrid (Dense + BM25 + Re-rank)`: Reciprocal Rank Fusion ($k=60$) combining keyword precision and semantic recall.
  - `Dense Semantic (FAISS)`: Cosine similarity via IndexFlatIP.
  - `Sparse Keyword (BM25)`: Okapi BM25 keyword matching.
- **Passage Inspection:** View retrieved excerpts with confidence scores, page numbers, and chunk IDs.

### Tab 3: Agentic Workflow & Review Synthesis
- **Input:** Enter an overarching research question (e.g., *"How do hybrid retrieval and agentic workflows improve factual consistency and reduce hallucinations in retrieval-augmented generation?"*).
- **Execution:** Click **"Run Complete Literature Review"**.
- **Live Stepper Tracker:** Displays active progress across all 11 stages:
  1. *Planning* $\rightarrow$ 2. *Retrieval* $\rightarrow$ 3. *Ingestion* $\rightarrow$ 4. *FAISS+BM25* $\rightarrow$ 5. *Extraction* $\rightarrow$ 6. *Clustering* $\rightarrow$ 7. *Comparison* $\rightarrow$ 8. *Gap Analysis* $\rightarrow$ 9. *Verification* $\rightarrow$ 10. *Synthesis* $\rightarrow$ 11. *Complete*.
- **Artifact Explorer:**
  - **Literature Review Report:** Full structured academic review with formatted sections, inline citations, copy button, and Markdown `.md` export.
  - **Comparison Matrix:** Interactive table comparing papers across Objective, Methodology, Dataset, Model, Metrics, Findings, and Limitations.
  - **Thematic Clusters:** Dense KMeans cluster cards with informative grounded topic titles.
  - **Research Gaps:** Grounded gap cards identifying limitations and future directions.
  - **Verification Audit:** Audit cards displaying verified claims with status pills (`Supported`, `Partially Supported`, `Unsupported`, `Conflicting`), grounding percentage, and source passage excerpts.
  - **Execution Plan:** Inspection of subtopic decomposition and evidence criteria.

---

## 8. Academic Evaluation & Demonstration Notes

For demonstration and grading in CSI4001:
1. **Factual Grounding:** The system explicitly distinguishes between extractable and scanned PDFs (marking scanned PDFs with an "OCR not implemented" notice) and flags missing fields as "Not reported in available text" rather than inventing data.
2. **Dual Mode Operation:** Full pipeline functionality is operational both with external LLM APIs (OpenAI) and in standalone deterministic offline mode.
3. **No Network Requirement for Testing:** The test suite mocks all network and embedding model calls, executing deterministically and reliably across all development platforms.
