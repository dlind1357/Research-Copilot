import re

# Approximation: 1 token ≈ 4 characters (standard GPT/Gemini heuristic)
CHARS_PER_TOKEN = 4
CHUNK_SIZE_TOKENS = 800
OVERLAP_TOKENS = 100

CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN   # 3200 chars
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN         # 400 chars


def _clean_text(text: str) -> str:
    """Remove excessive whitespace while preserving paragraph breaks."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of 3+ newlines into two (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs on a single line into one space
    text = re.sub(r"[ \t]+", " ", text)
    # Strip leading/trailing whitespace from each line
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks sized ~800 tokens with ~100-token overlap.

    Args:
        text: Raw extracted text from a PDF.

    Returns:
        List of text chunk strings.
    """
    if not text or not text.strip():
        return []

    text = _clean_text(text)
    text_length = len(text)

    if text_length <= CHUNK_SIZE_CHARS:
        return [text]

    chunks = []
    start = 0

    while start < text_length:
        end = start + CHUNK_SIZE_CHARS

        # Try to break at a sentence boundary (. ! ?) within the last 20% of the chunk
        if end < text_length:
            boundary_search_start = end - int(CHUNK_SIZE_CHARS * 0.2)
            match = None
            for m in re.finditer(r"[.!?]\s+", text[boundary_search_start:end]):
                match = m
            if match:
                end = boundary_search_start + match.end()

        chunks.append(text[start:end].strip())

        if end >= text_length:
            break

        start = end - OVERLAP_CHARS  # move back by overlap for context continuity

    return [c for c in chunks if c]  # filter any empty chunks
