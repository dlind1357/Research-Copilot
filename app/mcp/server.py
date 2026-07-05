import re
import logging
import httpx
from app.rag.retriever import retrieve_context
from app.tools.pubmed import search_pubmed
from app.tools.gene import lookup_gene

logger = logging.getLogger(__name__)

# ── Response size limits ───────────────────────────────────────────────────────
MAX_RAG_CHUNKS = 5
MAX_CHUNK_LENGTH = 1000      # chars per chunk

# ── Shared async HTTP client ───────────────────────────────────────────────────
_http_client = httpx.AsyncClient(timeout=15.0)


# ── Input helpers ──────────────────────────────────────────────────────────────

def _validate_text(value: str, field: str, max_len: int = 300) -> str:
    """Strip whitespace, reject empty inputs, and cap length."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} cannot be empty or whitespace-only.")
    if len(cleaned) > max_len:
        raise ValueError(f"{field} exceeds maximum length of {max_len} characters.")
    return cleaned


def _sanitize_text(text: str | None, max_len: int) -> str:
    """Remove control characters and truncate to max_len."""
    if not text:
        return ""
    # Strip non-printable control characters (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    if len(text) > max_len:
        text = text[:max_len] + "…"
    return text.strip()


# ── Tool 3: RAG search ────────────────────────────────────────────────────────

async def search_rag(query: str) -> dict:
    """Search the local vector store for relevant research paper chunks.

    Args:
        query: Natural language research question.

    Returns:
        Structured dict with ranked list of relevant text chunks and distances.
    """
    query = _validate_text(query, "query", max_len=300)

    try:
        results = retrieve_context(query, top_k=MAX_RAG_CHUNKS)
        chunks = [
            {
                "chunk_id": r["id"],
                "text": _sanitize_text(r["document"], MAX_CHUNK_LENGTH),
                "distance": round(r["distance"], 4),
                "metadata": r.get("metadata", {}),
            }
            for r in results
        ]
        return {
            "tool": "search_rag",
            "query": query,
            "count": len(chunks),
            "chunks": chunks,
        }

    except ValueError as e:
        raise
    except Exception as e:
        logger.error("RAG search failed. (Details hidden for security)")
        return {"tool": "search_rag", "query": query, "error": "RAG search unavailable.", "chunks": []}
