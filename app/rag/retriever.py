import logging
from app.rag.embeddings import EmbeddingService
from app.rag.vectorstore import search

logger = logging.getLogger(__name__)

# Shared embedding service instance
_embedding_service = EmbeddingService()


def retrieve_context(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve the most relevant text chunks for a given query.

    Flow:
      1. Validate and embed the query using EmbeddingService.
      2. Run cosine similarity search against ChromaDB.
      3. Return ranked list of relevant chunks with metadata.

    Args:
        query: The user's natural language question.
        top_k: Number of top results to return (default 5).

    Returns:
        List of dicts with keys: id, document, metadata, distance.

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or whitespace-only.")

    logger.info(f"Retrieving context for query ({len(query)} chars), top_k={top_k}")

    # Step 1: Embed the query
    query_embedding = _embedding_service.get_embedding(query)

    # Step 2: Search vector store
    results = search(query_embedding=query_embedding, top_k=top_k)

    logger.info(f"Retrieved {len(results)} chunks from vector store.")
    return results
