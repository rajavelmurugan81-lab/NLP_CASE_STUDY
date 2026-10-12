import os
import json
import sqlite3
from typing import List, Dict, Any, Optional

# Default database location if not provided in environment variables
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "literature_review.db")


def get_db_path() -> str:
    """Returns the resolved database path from environment or default."""
    return os.getenv("DATABASE_PATH", DEFAULT_DB_PATH)


def get_db_connection() -> sqlite3.Connection:
    """
    Establishes and returns a connection to the SQLite database.
    Rows are returned as sqlite3.Row objects to enable dictionary-like access.
    """
    db_path = get_db_path()
    
    # Ensure the parent directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initializes the SQLite database schema if the tables do not already exist.
    Applies non-destructive column migrations for academic paper metadata,
    workflow runs, extracted evidence, and literature reports.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table for storing user research search queries
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Table for storing metadata of retrieved academic papers
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER,
            title TEXT NOT NULL,
            authors TEXT,
            abstract TEXT,
            url TEXT,
            published_year INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_id) REFERENCES search_queries (id) ON DELETE CASCADE
        );
    """)

    # Check and add new columns for OpenAlex and citations if not already present
    cursor.execute("PRAGMA table_info(papers);")
    existing_columns = {col["name"] for col in cursor.fetchall()}

    if "openalex_id" not in existing_columns:
        cursor.execute("ALTER TABLE papers ADD COLUMN openalex_id TEXT;")
    if "doi" not in existing_columns:
        cursor.execute("ALTER TABLE papers ADD COLUMN doi TEXT;")
    if "pdf_url" not in existing_columns:
        cursor.execute("ALTER TABLE papers ADD COLUMN pdf_url TEXT;")
    if "citation_count" not in existing_columns:
        cursor.execute("ALTER TABLE papers ADD COLUMN citation_count INTEGER DEFAULT 0;")
    if "referenced_works_json" not in existing_columns:
        cursor.execute("ALTER TABLE papers ADD COLUMN referenced_works_json TEXT;")

    # Unique index for deduplication by openalex_id
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_openalex_id 
        ON papers (openalex_id) 
        WHERE openalex_id IS NOT NULL AND openalex_id != '';
    """)

    # Index for fast DOI lookup
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_papers_doi 
        ON papers (doi) 
        WHERE doi IS NOT NULL AND doi != '';
    """)

    # Junction table to associate queries with deduplicated papers
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_papers (
            query_id INTEGER,
            paper_id INTEGER,
            PRIMARY KEY (query_id, paper_id),
            FOREIGN KEY (query_id) REFERENCES search_queries (id) ON DELETE CASCADE,
            FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
        );
    """)

    # Table for bounded planning and iterative workflow runs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workflow_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            research_question TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            retry_count INTEGER DEFAULT 0,
            plan_json TEXT,
            results_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Table for structured academic evidence extraction
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS extracted_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER,
            openalex_id TEXT,
            objective TEXT,
            methodology TEXT,
            dataset TEXT,
            architecture TEXT,
            metrics TEXT,
            findings TEXT,
            limitations TEXT,
            future_work TEXT,
            evidence_passages_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
        );
    """)

    # Table for synthesized literature reports
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS literature_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_id INTEGER,
            research_question TEXT NOT NULL,
            report_markdown TEXT NOT NULL,
            report_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (workflow_id) REFERENCES workflow_runs (id) ON DELETE CASCADE
        );
    """)

    conn.commit()
    conn.close()


def save_search_query(topic: str, status: str = "completed") -> int:
    """
    Saves a user search topic in search_queries table and returns the generated query_id.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO search_queries (topic, status) VALUES (?, ?)",
        (topic, status)
    )
    conn.commit()
    query_id = cursor.lastrowid
    conn.close()
    return query_id


def save_or_get_paper(query_id: Optional[int], paper_data: dict) -> dict:
    """
    Saves a paper to the database or retrieves the existing record if already stored.
    Deduplicates using OpenAlex ID or normalized DOI.
    Associates the paper with query_id in query_papers.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    openalex_id = paper_data.get("openalex_id")
    doi = paper_data.get("doi")
    existing_paper = None

    # Check by OpenAlex ID
    if openalex_id:
        cursor.execute("SELECT * FROM papers WHERE openalex_id = ?", (openalex_id,))
        existing_paper = cursor.fetchone()

    # If not found by OpenAlex ID, check by DOI
    if not existing_paper and doi:
        cursor.execute("SELECT * FROM papers WHERE doi = ?", (doi,))
        existing_paper = cursor.fetchone()

    citation_count = paper_data.get("citation_count", 0)
    ref_works = paper_data.get("referenced_works") or paper_data.get("referenced_works_json")
    if isinstance(ref_works, list):
        ref_works_str = json.dumps(ref_works)
    elif isinstance(ref_works, str):
        ref_works_str = ref_works
    else:
        ref_works_str = None

    if existing_paper:
        paper_id = existing_paper["id"]
        # Update metadata if missing in existing record
        cursor.execute("""
            UPDATE papers SET
                abstract = COALESCE(abstract, ?),
                pdf_url = COALESCE(pdf_url, ?),
                url = COALESCE(url, ?),
                citation_count = CASE WHEN citation_count IS NULL OR citation_count = 0 THEN ? ELSE citation_count END,
                referenced_works_json = COALESCE(referenced_works_json, ?)
            WHERE id = ?
        """, (
            paper_data.get("abstract"),
            paper_data.get("pdf_url"),
            paper_data.get("url"),
            citation_count,
            ref_works_str,
            paper_id
        ))
    else:
        # Insert new paper record
        cursor.execute("""
            INSERT INTO papers (
                query_id, openalex_id, doi, title, authors,
                abstract, url, pdf_url, published_year,
                citation_count, referenced_works_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query_id,
            openalex_id,
            doi,
            paper_data.get("title", "Untitled Document"),
            paper_data.get("authors", "Unknown Authors"),
            paper_data.get("abstract"),
            paper_data.get("url"),
            paper_data.get("pdf_url"),
            paper_data.get("published_year"),
            citation_count,
            ref_works_str
        ))
        paper_id = cursor.lastrowid

    # Link to current query in query_papers
    if query_id:
        cursor.execute("""
            INSERT OR IGNORE INTO query_papers (query_id, paper_id)
            VALUES (?, ?)
        """, (query_id, paper_id))

    conn.commit()

    # Fetch and return complete stored record
    cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
    saved_record = dict(cursor.fetchone())
    conn.close()

    return saved_record


def get_papers_by_query_id(query_id: int) -> List[Dict[str, Any]]:
    """Retrieves all papers associated with a query_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.* FROM papers p
        JOIN query_papers qp ON p.id = qp.paper_id
        WHERE qp.query_id = ?
        ORDER BY p.id ASC
    """, (query_id,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_all_papers() -> List[Dict[str, Any]]:
    """Retrieves all unique papers in the SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM papers ORDER BY id DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ==============================================================
# Workflow Runs & Persistence Functions
# ==============================================================

def create_workflow_run(research_question: str, plan_data: Optional[dict] = None) -> int:
    """Initializes a new workflow run in SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    plan_str = json.dumps(plan_data) if plan_data else None
    cursor.execute(
        "INSERT INTO workflow_runs (research_question, status, plan_json) VALUES (?, 'planning', ?)",
        (research_question, plan_str)
    )
    conn.commit()
    workflow_id = cursor.lastrowid
    conn.close()
    return workflow_id


def update_workflow_run(workflow_id: int, status: str, results_data: Optional[dict] = None, retry_count: Optional[int] = None) -> None:
    """Updates the status and intermediate results of a workflow run."""
    conn = get_db_connection()
    cursor = conn.cursor()
    results_str = json.dumps(results_data) if results_data else None
    
    if retry_count is not None and results_str is not None:
        cursor.execute(
            "UPDATE workflow_runs SET status = ?, results_json = ?, retry_count = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, results_str, retry_count, workflow_id)
        )
    elif results_str is not None:
        cursor.execute(
            "UPDATE workflow_runs SET status = ?, results_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, results_str, workflow_id)
        )
    else:
        cursor.execute(
            "UPDATE workflow_runs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, workflow_id)
        )
    conn.commit()
    conn.close()


def get_workflow_run(workflow_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a workflow run record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM workflow_runs WHERE id = ?", (workflow_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    if data.get("plan_json"):
        try:
            data["plan"] = json.loads(data["plan_json"])
        except Exception:
            data["plan"] = None
    if data.get("results_json"):
        try:
            data["results"] = json.loads(data["results_json"])
        except Exception:
            data["results"] = None
    return data


def save_extracted_evidence(evidence_data: dict) -> int:
    """Saves structured evidence extracted from a paper."""
    conn = get_db_connection()
    cursor = conn.cursor()
    passages_str = json.dumps(evidence_data.get("evidence_passages", []))
    cursor.execute("""
        INSERT INTO extracted_evidence (
            paper_id, openalex_id, objective, methodology, dataset,
            architecture, metrics, findings, limitations, future_work,
            evidence_passages_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        evidence_data.get("paper_id"),
        evidence_data.get("openalex_id"),
        evidence_data.get("objective", "Unknown"),
        evidence_data.get("methodology", "Unknown"),
        evidence_data.get("dataset", "Unknown"),
        evidence_data.get("architecture", "Unknown"),
        evidence_data.get("metrics", "Unknown"),
        evidence_data.get("findings", "Unknown"),
        evidence_data.get("limitations", "Unknown"),
        evidence_data.get("future_work", "Unknown"),
        passages_str
    ))
    conn.commit()
    eid = cursor.lastrowid
    conn.close()
    return eid


def get_extracted_evidence_by_paper_id(paper_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves extracted evidence for a specific paper."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM extracted_evidence WHERE paper_id = ? ORDER BY id DESC LIMIT 1", (paper_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    if data.get("evidence_passages_json"):
        try:
            data["evidence_passages"] = json.loads(data["evidence_passages_json"])
        except Exception:
            data["evidence_passages"] = []
    return data


def save_literature_report(workflow_id: int, research_question: str, report_markdown: str, report_json: Optional[dict] = None) -> int:
    """Saves a synthesized literature review report."""
    conn = get_db_connection()
    cursor = conn.cursor()
    report_json_str = json.dumps(report_json) if report_json else None
    cursor.execute("""
        INSERT INTO literature_reports (
            workflow_id, research_question, report_markdown, report_json
        ) VALUES (?, ?, ?, ?)
    """, (workflow_id, research_question, report_markdown, report_json_str))
    conn.commit()
    report_id = cursor.lastrowid
    conn.close()
    return report_id


def get_latest_report(workflow_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the latest literature report."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if workflow_id:
        cursor.execute("SELECT * FROM literature_reports WHERE workflow_id = ? ORDER BY id DESC LIMIT 1", (workflow_id,))
    else:
        cursor.execute("SELECT * FROM literature_reports ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    if data.get("report_json"):
        try:
            data["report_data"] = json.loads(data["report_json"])
        except Exception:
            data["report_data"] = None
    return data
