import os
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from database.db import (
    init_db,
    get_all_papers,
    get_workflow_run,
    get_latest_report
)
from agents.retrieval_agent import RetrievalAgent
from agents.rag_agent import RagAgent
from agents.planning_agent import PlanningAgent
from agents.evidence_extraction_agent import EvidenceExtractionAgent
from agents.comparative_agent import ComparativeAgent
from agents.gap_analysis_agent import ResearchGapAgent
from agents.verification_agent import CitationVerificationAgent
from agents.workflow_coordinator import WorkflowCoordinator
from services.hybrid_retriever import HybridRetriever
from services.clustering_service import cluster_papers
from services.llm_service import LLMService
from services.openalex_service import OpenAlexConnectionError, OpenAlexAPIError
from services.pdf_service import (
    safe_resolve_pdf_path,
    PDFProcessingError,
    PDFEncryptedError,
    PDFCorruptedError,
    DEFAULT_UPLOAD_DIR
)
from services.embedding_service import EmbeddingError, ModelLoadError
from services.vector_store import VectorStoreError

# Load environment variables
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "college-nlp-secret-key-default")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 26214400))  # 25MB
app.config["UPLOAD_FOLDER"] = DEFAULT_UPLOAD_DIR

# Ensure uploads directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize database schema on startup
with app.app_context():
    init_db()

# Lazy singletons for agent pipeline components
_rag_agent_instance = None
_hybrid_retriever_instance = None
_workflow_coordinator_instance = None


def get_rag_agent() -> RagAgent:
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = RagAgent()
    return _rag_agent_instance


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever_instance
    if _hybrid_retriever_instance is None:
        rag = get_rag_agent()
        _hybrid_retriever_instance = HybridRetriever(
            vector_store=rag.vector_store,
            embedding_service=rag.embedding_service
        )
    return _hybrid_retriever_instance


def get_workflow_coordinator() -> WorkflowCoordinator:
    global _workflow_coordinator_instance
    if _workflow_coordinator_instance is None:
        rag = get_rag_agent()
        hybrid = get_hybrid_retriever()
        _workflow_coordinator_instance = WorkflowCoordinator(
            rag_agent=rag,
            hybrid_retriever=hybrid
        )
    return _workflow_coordinator_instance


@app.route("/")
def index():
    """Renders the academic research dashboard frontend."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint returning server status.
    Confirms that the Flask backend is active and responding.
    """
    return jsonify({
        "status": "healthy",
        "service": "Literature Review Agent Backend",
        "database": "sqlite_ready",
        "message": "Server is running successfully."
    }), 200


@app.route("/api/llm/status", methods=["GET"])
def llm_status():
    """Checks whether an external LLM API key is configured."""
    llm = LLMService()
    configured = llm.is_configured()
    return jsonify({
        "status": "success",
        "configured": configured,
        "provider": "OpenAI" if configured else "Offline Heuristic Fallback",
        "model": llm.model if configured else "Heuristic / Rule-based NLP",
        "message": "LLM configured." if configured else "OPENAI_API_KEY is not set or uses a placeholder. System operating in deterministic heuristic mode."
    }), 200


@app.route("/api/search", methods=["POST"])
def search_papers():
    """
    Real academic paper search endpoint via OpenAlex and RetrievalAgent.
    Normalizes metadata, saves search query, and deduplicates papers in SQLite.
    """
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "").strip()
    raw_limit = data.get("limit", 10)

    if not topic:
        return jsonify({
            "status": "error",
            "message": "Please enter a valid research topic."
        }), 400

    try:
        limit = int(raw_limit)
    except (ValueError, TypeError):
        limit = 10

    try:
        agent = RetrievalAgent(default_limit=10)
        results = agent.retrieve_papers(query=topic, limit=limit, save_to_db=True)
        return jsonify(results), 200

    except OpenAlexConnectionError as conn_err:
        return jsonify({
            "status": "error",
            "message": f"Unable to reach OpenAlex academic repository: {str(conn_err)}"
        }), 502

    except OpenAlexAPIError as api_err:
        return jsonify({
            "status": "error",
            "message": f"OpenAlex API error: {str(api_err)}"
        }), 502

    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": f"An unexpected error occurred while retrieving papers: {str(exc)}"
        }), 500


@app.route("/api/upload", methods=["POST"])
def upload_pdfs():
    """
    Uploads one or multiple research PDF files.
    Enforces safe filenames, file extension checks, and size validation.
    """
    uploaded_files = request.files.getlist("files") or request.files.getlist("file")

    if not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == ""):
        return jsonify({
            "status": "error",
            "message": "No PDF files provided in the upload request."
        }), 400

    saved_records = []
    errors = []

    for file_obj in uploaded_files:
        if not file_obj or not file_obj.filename:
            continue

        filename = secure_filename(file_obj.filename)
        if not filename.lower().endswith(".pdf"):
            errors.append(f"Rejected '{file_obj.filename}': Only .pdf files are accepted.")
            continue

        target_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            file_obj.save(target_path)
            file_size = os.path.getsize(target_path)
            saved_records.append({
                "filename": filename,
                "filepath": target_path,
                "size_bytes": file_size
            })
        except Exception as exc:
            errors.append(f"Failed to save '{filename}': {str(exc)}")

    if not saved_records and errors:
        return jsonify({
            "status": "error",
            "message": "Upload failed.",
            "errors": errors
        }), 400

    return jsonify({
        "status": "success",
        "count": len(saved_records),
        "files": saved_records,
        "errors": errors if errors else None,
        "message": f"Successfully uploaded {len(saved_records)} PDF document(s)."
    }), 201


@app.route("/api/ingest", methods=["POST"])
def ingest_pdf():
    """
    Ingests an existing accessible PDF from the uploads folder into the FAISS vector knowledge base.
    Restricts filesystem access strictly to the uploads directory.
    """
    data = request.get_json(silent=True) or {}
    filename = data.get("filename") or data.get("filepath")
    paper_id = data.get("paper_id")
    doi = data.get("doi")
    source_url = data.get("source_url")

    if not filename:
        return jsonify({
            "status": "error",
            "message": "Missing 'filename' parameter for PDF ingestion."
        }), 400

    try:
        safe_path = safe_resolve_pdf_path(filename, allowed_dir=app.config["UPLOAD_FOLDER"])
    except PDFProcessingError as proc_err:
        return jsonify({
            "status": "error",
            "message": str(proc_err)
        }), 400

    try:
        rag_agent = get_rag_agent()
        result = rag_agent.ingest_pdf(
            pdf_path=safe_path,
            paper_id=paper_id,
            filename=os.path.basename(safe_path),
            doi=doi,
            source_url=source_url
        )
        status_code = 200 if result.get("status") in ("success", "already_indexed") else 422
        return jsonify(result), status_code

    except (PDFEncryptedError, PDFCorruptedError) as pdf_err:
        return jsonify({
            "status": "error",
            "message": str(pdf_err)
        }), 422
    except ModelLoadError as mdl_err:
        return jsonify({
            "status": "error",
            "message": f"Embedding model unavailable: {str(mdl_err)}"
        }), 503
    except (EmbeddingError, VectorStoreError) as rag_err:
        return jsonify({
            "status": "error",
            "message": f"RAG ingestion failure: {str(rag_err)}"
        }), 500
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": f"Unexpected ingestion error: {str(exc)}"
        }), 500


@app.route("/api/retrieve", methods=["POST"])
def retrieve_passages():
    """
    Retrieves the top relevant evidence passages from the indexed literature for a research query (Dense).
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    raw_top_k = data.get("top_k", 5)

    if not query:
        return jsonify({
            "status": "error",
            "message": "Please provide a valid research question or query."
        }), 400

    try:
        top_k = max(1, min(int(raw_top_k), 20))
    except (ValueError, TypeError):
        top_k = 5

    try:
        rag_agent = get_rag_agent()
        results = rag_agent.retrieve_passages(query=query, top_k=top_k)
        return jsonify(results), 200

    except ModelLoadError as mdl_err:
        return jsonify({
            "status": "error",
            "message": f"Embedding model unavailable: {str(mdl_err)}"
        }), 503
    except (EmbeddingError, VectorStoreError) as rag_err:
        return jsonify({
            "status": "error",
            "message": f"Vector retrieval error: {str(rag_err)}"
        }), 500
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": f"Unexpected retrieval error: {str(exc)}"
        }), 500


@app.route("/api/retrieve/hybrid", methods=["POST"])
def retrieve_hybrid_passages():
    """
    Hybrid Retrieval endpoint combining BM25 keyword matching and FAISS dense similarity with RRF and re-ranking.
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"status": "error", "message": "Query string cannot be empty."}), 400

    try:
        hybrid = get_hybrid_retriever()
        results = hybrid.retrieve_hybrid(query=query, top_k=top_k, rerank=True)
        return jsonify({
            "status": "success",
            "query": query,
            "count": len(results),
            "passages": results,
            "fusion_method": "Reciprocal Rank Fusion (k=60) + Semantic Re-ranking"
        }), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/plan", methods=["POST"])
def plan_research():
    """Generates research subtopics, search queries, and evidence criteria for a question."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"status": "error", "message": "Research question is required."}), 400

    try:
        planner = PlanningAgent()
        plan = planner.plan_research(question)
        return jsonify({"status": "success", "plan": plan}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/cluster", methods=["POST"])
def cluster_endpoint():
    """Groups papers into thematic clusters based on dense representation similarities."""
    data = request.get_json(silent=True) or {}
    papers = data.get("papers") or get_all_papers()
    num_clusters = int(data.get("num_clusters", 3))

    try:
        rag = get_rag_agent()
        clusters = cluster_papers(papers=papers, embedding_service=rag.embedding_service, num_clusters=num_clusters)
        return jsonify({"status": "success", "data": clusters}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/compare", methods=["POST"])
def compare_endpoint():
    """Generates cross-paper comparison matrix and empirical synthesis."""
    data = request.get_json(silent=True) or {}
    evidence_list = data.get("evidence") or []
    if not evidence_list:
        papers = get_all_papers()[:6]
        extractor = EvidenceExtractionAgent()
        evidence_list = [extractor.extract_evidence(p, save_to_db=False) for p in papers]

    try:
        comparator = ComparativeAgent()
        comp_res = comparator.compare_papers(evidence_list)
        return jsonify({"status": "success", "data": comp_res}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/gaps", methods=["POST"])
def gaps_endpoint():
    """Identifies evidence-supported research gaps across reviewed papers."""
    data = request.get_json(silent=True) or {}
    evidence_list = data.get("evidence") or []
    if not evidence_list:
        papers = get_all_papers()[:6]
        extractor = EvidenceExtractionAgent()
        evidence_list = [extractor.extract_evidence(p, save_to_db=False) for p in papers]

    try:
        gap_agent = ResearchGapAgent()
        gaps_res = gap_agent.identify_gaps(evidence_list)
        return jsonify({"status": "success", "data": gaps_res}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/verify", methods=["POST"])
def verify_endpoint():
    """Verifies synthesized claims against retrieved evidence passages and source papers."""
    data = request.get_json(silent=True) or {}
    claims = data.get("claims") or []
    known_papers = data.get("papers") or get_all_papers()
    passages = data.get("passages") or []

    try:
        rag = get_rag_agent()
        verifier = CitationVerificationAgent(embedding_service=rag.embedding_service)
        result = verifier.verify_claims(claims, known_papers, passages)
        return jsonify({"status": "success", "data": result}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/workflow/run", methods=["POST"])
def run_workflow_endpoint():
    """
    Executes the complete 11-stage autonomous agentic literature review pipeline.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"status": "error", "message": "Research question is required."}), 400

    try:
        coordinator = get_workflow_coordinator()
        workflow_res = coordinator.run_full_review(research_question=question)
        return jsonify({"status": "success", "data": workflow_res}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Workflow execution error: {str(exc)}"}), 500


@app.route("/api/workflow/<int:workflow_id>", methods=["GET"])
def get_workflow_status(workflow_id: int):
    """Retrieves the status and intermediate results of a workflow run."""
    run = get_workflow_run(workflow_id)
    if not run:
        return jsonify({"status": "error", "message": "Workflow run not found."}), 404
    return jsonify({"status": "success", "data": run}), 200


@app.route("/api/report/latest", methods=["GET"])
def get_latest_report_endpoint():
    """Returns the most recently synthesized literature review report."""
    report = get_latest_report()
    if not report:
        return jsonify({"status": "empty", "message": "No reports generated yet."}), 404
    return jsonify({"status": "success", "data": report}), 200


@app.route("/api/report/export", methods=["GET"])
def export_report_markdown():
    """Downloads the latest literature review report as a Markdown (.md) file."""
    report = get_latest_report()
    if not report:
        return jsonify({"status": "error", "message": "No report available to export."}), 404

    filename = f"literature_review_{report.get('id', 1)}.md"
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(report.get("report_markdown", ""))

    return send_file(temp_path, as_attachment=True, download_name=filename)
@app.route("/api/rag/stats", methods=["GET"])
def rag_stats():
    """Returns knowledge base vector count and indexed document statistics."""
    try:
        rag_agent = get_rag_agent()
        return jsonify({
            "status": "success",
            "stats": rag_agent.vector_store.get_stats()
        }), 200
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    print(f"Starting Literature Review Flask server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=debug)
