import re
from typing import List, Dict, Any, Optional

DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 120


def clean_text(text: str) -> str:
    """
    Cleans extracted PDF text while preserving scientific notation,
    formulas, citation brackets, and chemical/mathematical symbols.
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize carriage returns and non-breaking spaces
    cleaned = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")

    # Reconnect words split across lines by hyphens (e.g. "con-\ntext" -> "context")
    cleaned = re.sub(r'(\b\w+)-\n(\w+\b)', r'\1\2', cleaned)

    # Collapse excessive line breaks into paragraphs
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    # Collapse multiple inline spaces and tabs
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)

    # Remove trailing/leading spaces on lines
    cleaned = re.sub(r' *\n *', '\n', cleaned)

    return cleaned.strip()


def find_best_break_point(text: str, start: int, target_end: int) -> int:
    """
    Finds a natural sentence or paragraph boundary near target_end to prevent
    awkward splits inside words or formulas.
    """
    if target_end >= len(text):
        return len(text)

    # Look backwards within a window of up to 80 chars for clean separators
    search_window_start = max(start + 50, target_end - 80)
    window = text[search_window_start:target_end + 20]

    # Priority 1: Paragraph break
    para_break = window.rfind("\n\n")
    if para_break != -1:
        return search_window_start + para_break + 2

    # Priority 2: Sentence terminal (. ! ?) followed by space or newline
    sentence_matches = list(re.finditer(r'[\.\?!]\s', window))
    if sentence_matches:
        last_match = sentence_matches[-1]
        return search_window_start + last_match.end()

    # Priority 3: Semicolon, colon, or newline
    semi_matches = list(re.finditer(r'[\n;:]\s*', window))
    if semi_matches:
        return search_window_start + semi_matches[-1].end()

    # Priority 4: Word boundary (space)
    space_pos = window.rfind(" ")
    if space_pos != -1:
        return search_window_start + space_pos + 1

    return target_end


def split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Splits text into overlapping chunks using sliding window with natural boundary detection.
    """
    if not text:
        return []

    text_len = len(text)
    if text_len <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < text_len:
        target_end = min(start + chunk_size, text_len)
        actual_end = find_best_break_point(text, start, target_end)

        chunk = text[start:actual_end].strip()
        if chunk:
            chunks.append(chunk)

        if actual_end >= text_len:
            break

        # Calculate next start with overlap, guaranteeing forward progress
        min_progress = max(1, chunk_size - chunk_overlap)
        next_start = actual_end - chunk_overlap
        if next_start <= start:
            next_start = start + min_progress

        start = next_start

    return chunks


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    paper_metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Chunks a list of extracted PDF page dictionaries.
    Maintains clean page number attribution and source document identifiers for every chunk.
    Avoids mixing source metadata between papers.
    """
    paper_metadata = paper_metadata or {}
    paper_id = paper_metadata.get("paper_id")
    filename = paper_metadata.get("filename", "unknown_document.pdf")
    doi = paper_metadata.get("doi")
    source_url = paper_metadata.get("source_url")

    all_chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for page_info in pages:
        page_num = page_info.get("page_number", 1)
        raw_page_text = page_info.get("text", "")
        cleaned_page_text = clean_text(raw_page_text)

        if not cleaned_page_text:
            continue

        page_chunks = split_text_into_chunks(
            cleaned_page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for chunk_text in page_chunks:
            chunk_counter += 1
            # Unique stable identifier for this chunk
            doc_prefix = f"doc_{paper_id}" if paper_id else re.sub(r'[^a-zA-Z0-9]', '_', filename)
            chunk_id = f"{doc_prefix}_p{page_num}_c{chunk_counter}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk_counter,
                "page_number": page_num,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "paper_id": paper_id,
                "filename": filename,
                "doi": doi,
                "source_url": source_url
            })

    return all_chunks
