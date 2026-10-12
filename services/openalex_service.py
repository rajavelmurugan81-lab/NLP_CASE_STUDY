import os
import requests
from typing import List, Dict, Any, Optional

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_TIMEOUT = 12


class OpenAlexError(Exception):
    """Base exception for OpenAlex service errors."""
    pass


class OpenAlexConnectionError(OpenAlexError):
    """Raised when network connection fails or times out."""
    pass


class OpenAlexAPIError(OpenAlexError):
    """Raised when OpenAlex API returns an HTTP error status."""
    pass


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """
    Reconstructs an academic abstract from OpenAlex inverted-index representation.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return None

    pos_to_word: Dict[int, str] = {}
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, int):
                    pos_to_word[pos] = word

    if not pos_to_word:
        return None

    max_pos = max(pos_to_word.keys())
    ordered_words = [pos_to_word.get(i, "") for i in range(max_pos + 1)]
    reconstructed = " ".join(" ".join(ordered_words).split()).strip()

    return reconstructed if reconstructed else None


def _get_headers_and_auth() -> tuple[dict, dict]:
    """Constructs headers and query parameters for OpenAlex authentication."""
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    email = os.getenv("OPENALEX_EMAIL", "").strip()

    headers = {
        "Accept": "application/json",
        "User-Agent": f"LitReviewAI/1.0 (mailto:{email})" if email else "LitReviewAI/1.0 (Academic Research Assistant)"
    }
    params: Dict[str, Any] = {}
    if api_key:
        headers["api_key"] = api_key
        params["api_key"] = api_key
    if email:
        params["mailto"] = email
    return headers, params


def search_works(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Searches the OpenAlex academic API for scholarly works matching the query.
    """
    query_clean = query.strip()
    if not query_clean:
        return []

    clamped_limit = max(1, min(int(limit), 50))
    headers, params = _get_headers_and_auth()
    params["search"] = query_clean
    params["per_page"] = clamped_limit

    try:
        response = requests.get(
            OPENALEX_WORKS_URL,
            headers=headers,
            params=params,
            timeout=DEFAULT_TIMEOUT
        )
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as conn_err:
        raise OpenAlexConnectionError(f"Network error connecting to OpenAlex API: {str(conn_err)}") from conn_err
    except requests.exceptions.RequestException as req_err:
        raise OpenAlexConnectionError(f"Request failed: {str(req_err)}") from req_err

    if response.status_code != 200:
        raise OpenAlexAPIError(
            f"OpenAlex API responded with HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        data = response.json()
    except ValueError as json_err:
        raise OpenAlexAPIError("Failed to parse JSON response from OpenAlex.") from json_err

    results = data.get("results", [])
    return results if isinstance(results, list) else []


def fetch_work_by_id(work_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches a single academic work by its OpenAlex ID or DOI.
    """
    if not work_id or not str(work_id).strip():
        return None

    clean_id = str(work_id).strip()
    # Normalize ID if it has full URL
    if clean_id.startswith("https://openalex.org/"):
        item_id = clean_id.replace("https://openalex.org/", "")
    else:
        item_id = clean_id

    url = f"{OPENALEX_WORKS_URL}/{item_id}"
    headers, params = _get_headers_and_auth()

    try:
        response = requests.get(url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None
