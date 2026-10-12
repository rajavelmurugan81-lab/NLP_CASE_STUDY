"""
database package:
Handles SQLite database connection lifecycle, schema setup, paper persistence, workflow records, evidence, and reports.
"""
from .db import (
    get_db_connection,
    init_db,
    save_search_query,
    save_or_get_paper,
    get_papers_by_query_id,
    get_all_papers,
    create_workflow_run,
    update_workflow_run,
    get_workflow_run,
    save_extracted_evidence,
    get_extracted_evidence_by_paper_id,
    save_literature_report,
    get_latest_report
)

__all__ = [
    "get_db_connection",
    "init_db",
    "save_search_query",
    "save_or_get_paper",
    "get_papers_by_query_id",
    "get_all_papers",
    "create_workflow_run",
    "update_workflow_run",
    "get_workflow_run",
    "save_extracted_evidence",
    "get_extracted_evidence_by_paper_id",
    "save_literature_report",
    "get_latest_report"
]
