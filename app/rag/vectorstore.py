import logging
import uuid
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Collection name for scientific papers
COLLECTION_NAME = "research_papers"

_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
        
    from app.config.settings import settings
    from app.tools.gcs import sync_gcs_to_local
    
    if settings.STORAGE_TYPE == "local":
        path = settings.LOCAL_CHROMA_PATH
    else:
        path = "./chroma_db_gcs_cache"
        try:
            logger.info(f"Syncing ChromaDB from GCS ({settings.GCS_CHROMA_PATH}) to local cache...")
            sync_gcs_to_local(settings.GCS_CHROMA_PATH, path)
        except Exception as e:
            logger.warning(f"Could not sync GCS ChromaDB index to local cache (may be initializing a new database): {e}")
            
    _client = chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False)
    )
    return _client


def is_pmid_stored(pmid: str) -> bool:
    """Check if any document with the given PMID is already stored in ChromaDB."""
    if not pmid:
        return False
    try:
        collection = _get_collection()
        res = collection.get(where={"pmid": str(pmid)}, limit=1)
        return bool(res and res.get("ids"))
    except Exception as e:
        logger.warning(f"Error checking if PMID {pmid} is stored: {e}")
        return False


def get_stored_abstract(pmid: str) -> str | None:
    """Retrieve abstract/text for a given PMID from ChromaDB if it exists."""
    if not pmid:
        return None
    try:
        collection = _get_collection()
        res = collection.get(where={"pmid": str(pmid), "section": "Abstract"})
        if res and res.get("documents"):
            return "\n".join(res["documents"])
    except Exception as e:
        logger.warning(f"Error retrieving abstract for PMID {pmid} from ChromaDB: {e}")
    return None


def is_pdf_stored(pdf_filename: str) -> bool:
    """Check if any document with the given PDF filename is already stored in ChromaDB."""
    if not pdf_filename:
        return False
    try:
        collection = _get_collection()
        res = collection.get(where={"pdf_filename": str(pdf_filename)}, limit=1)
        return bool(res and res.get("ids"))
    except Exception as e:
        logger.warning(f"Error checking if PDF {pdf_filename} is stored: {e}")
        return False


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection with cosine similarity."""
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )



def add_documents(
    chunks: list[str],
    embeddings: list[list[float]],
    metadata: list[dict] | None = None
) -> list[str]:
    """Store text chunks and their embeddings in ChromaDB.

    Args:
        chunks: List of text chunks to store.
        embeddings: List of embedding vectors, one per chunk.
        metadata: Optional list of metadata dicts, one per chunk.

    Returns:
        List of document IDs assigned to the stored chunks.

    Raises:
        ValueError: If chunks and embeddings lengths do not match.
    """
    if not chunks:
        raise ValueError("chunks list cannot be empty.")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have the same length."
        )

    collection = _get_collection()

    # Generate unique IDs for each chunk
    doc_ids = [str(uuid.uuid4()) for _ in chunks]

    # Fall back to empty metadata dicts if not provided
    if metadata is None:
        metadata = [{} for _ in chunks]
    elif len(metadata) != len(chunks):
        raise ValueError(
            f"metadata ({len(metadata)}) and chunks ({len(chunks)}) must have the same length."
        )

    collection.add(
        ids=doc_ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
    )

    logger.info(f"Stored {len(doc_ids)} chunks in ChromaDB collection '{COLLECTION_NAME}'.")

    # Sync back to GCS if in gcs mode
    from app.config.settings import settings
    from app.tools.gcs import sync_local_to_gcs
    if settings.STORAGE_TYPE == "gcs":
        try:
            logger.info(f"Syncing updated ChromaDB cache to GCS ({settings.GCS_CHROMA_PATH})...")
            sync_local_to_gcs("./chroma_db_gcs_cache", settings.GCS_CHROMA_PATH)
            logger.info("Successfully synced ChromaDB cache to GCS.")
        except Exception as e:
            logger.error(f"Failed to sync ChromaDB cache back to GCS: {e}")

    return doc_ids


def search(
    query_embedding: list[float],
    top_k: int = 5,
    where: dict | None = None
) -> list[dict]:
    """Find the top-k most similar chunks for a given query embedding.

    Args:
        query_embedding: Embedding vector for the query.
        top_k: Number of results to return.
        where: Optional ChromaDB metadata filter dict.

    Returns:
        List of result dicts with keys: id, document, metadata, distance.
    """
    if not query_embedding:
        raise ValueError("query_embedding cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    collection = _get_collection()

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    # Flatten ChromaDB's nested result lists into a clean list of dicts
    output = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        output.append({
            "id": doc_id,
            "document": doc,
            "metadata": meta,
            "distance": dist,
        })

    return output


def delete_collection() -> None:
    """Delete the entire collection (useful for testing / resets)."""
    _get_client().delete_collection(COLLECTION_NAME)
    logger.info(f"Deleted ChromaDB collection '{COLLECTION_NAME}'.")
    
    # Sync deletion back to GCS if in gcs mode
    from app.config.settings import settings
    from app.tools.gcs import sync_local_to_gcs
    if settings.STORAGE_TYPE == "gcs":
        try:
            sync_local_to_gcs("./chroma_db_gcs_cache", settings.GCS_CHROMA_PATH)
        except Exception as e:
            logger.error(f"Failed to sync deleted collection back to GCS: {e}")


def count_documents() -> int:
    """Return the total number of documents stored in the collection."""
    return _get_collection().count()
