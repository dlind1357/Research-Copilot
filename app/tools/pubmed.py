"""PubMed search tool using NCBI E-utilities API."""

import re
import logging
import asyncio
import xml.etree.ElementTree as ET
import httpx
from app.rag.chunker import chunk_text
from app.rag.embeddings import EmbeddingService
from app.rag.vectorstore import add_documents

logger = logging.getLogger(__name__)

ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_RESULTS = 5
TIMEOUT_SECONDS = 15.0


def _sanitize(text: str | None, max_len: int = 300) -> str:
    """Strip control characters and truncate."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(text))
    return (text[:max_len] + "…") if len(text) > max_len else text.strip()


def _clean_and_sanitize_text(text: str) -> str:
    """Strip remaining XML/HTML tags and normalize whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return cleaned.strip()


async def _robust_get(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    """Make an HTTP GET request with retries on 429 rate limiting."""
    for attempt in range(4):
        try:
            r = await client.get(url, params=params)
            if r.status_code == 429:
                backoff = 1.5 * (attempt + 1)
                logger.warning(f"NCBI rate limit (429) hit. Backing off for {backoff}s...")
                await asyncio.sleep(backoff)
                continue
            return r
        except Exception as e:
            if attempt == 3:
                raise
            backoff = 1.5 * (attempt + 1)
            logger.warning(f"NCBI transport exception: {e}. Backing off for {backoff}s...")
            await asyncio.sleep(backoff)
    return await client.get(url, params=params)


def _parse_pmc_xml(xml_content: bytes) -> tuple[str, list[dict]]:
    """Parse article abstract and sections from PMC XML.
    
    Returns (abstract_text, list of dicts with [{'title': '...', 'text': '...'}]).
    """
    root = ET.fromstring(xml_content)
    
    # 1. Extract abstract
    abstract_p_els = root.findall(".//abstract//p")
    abstract_text = ""
    if abstract_p_els:
        text_parts = ["".join(p.itertext()).strip() for p in abstract_p_els]
        abstract_text = "\n".join([tp for tp in text_parts if tp])
    
    sections = []
    
    # 2. Extract sections
    sec_elements = root.findall(".//sec")
    if sec_elements:
        for sec in sec_elements:
            title_el = sec.find("title")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""
            p_els = sec.findall(".//p")
            if p_els:
                text_parts = ["".join(p.itertext()).strip() for p in p_els]
                text = "\n".join([tp for tp in text_parts if tp])
                if text:
                    sections.append({
                        "title": title,
                        "text": text
                    })
                    
    # Fallback if no sections were extracted but abstract or paragraphs exist
    if not sections:
        if abstract_text:
            sections.append({
                "title": "Abstract",
                "text": abstract_text
            })
        else:
            p_els = root.findall(".//p")
            if p_els:
                text_parts = ["".join(p.itertext()).strip() for p in p_els]
                text = "\n".join([tp for tp in text_parts if tp])
                if text:
                    sections.append({
                        "title": "Full Text",
                        "text": text
                    })
                    
    return abstract_text, sections


async def _fetch_abstract_if_available(client: httpx.AsyncClient, pmid: str) -> str:
    """Fetch abstract via efetch if available, returning empty string if not found or on error."""
    try:
        r = await _robust_get(
            client,
            f"{ENTREZ_BASE}/efetch.fcgi",
            params={"db": "pubmed", "id": pmid, "retmode": "xml"}
        )
        if r.status_code != 200:
            return ""
        root = ET.fromstring(r.content)
        abstract_texts = root.findall(".//AbstractText")
        if not abstract_texts:
            return ""
        text_pieces = ["".join(t.itertext()).strip() for t in abstract_texts if t is not None]
        return " ".join([tp for tp in text_pieces if tp])
    except Exception as e:
        logger.warning(f"Error fetching abstract for PMID {pmid}: {e}")
        return ""


async def _process_and_store_pmc(client: httpx.AsyncClient, pmid: str, pmcid: str, title: str) -> str:
    """Download full PMC XML, parse sections, sanitize, chunk, embed, and store in ChromaDB."""
    try:
        logger.info(f"Downloading full text PMC XML for {pmcid} (PMID: {pmid})...")
        r = await _robust_get(
            client,
            f"{ENTREZ_BASE}/efetch.fcgi",
            params={"db": "pmc", "id": pmcid, "retmode": "xml"}
        )
        if r.status_code != 200:
            logger.warning(f"Failed to fetch PMC XML for {pmcid}: status {r.status_code}")
            return ""
            
        abstract_text, sections = _parse_pmc_xml(r.content)
        
        all_chunks = []
        all_metadatas = []
        
        for sec in sections:
            sec_title = sec["title"]
            sec_text = _clean_and_sanitize_text(sec["text"])
            if not sec_text:
                continue
                
            chunks = chunk_text(sec_text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({
                    "pmcid": pmcid,
                    "pmid": pmid,
                    "title": title,
                    "section": sec_title,
                    "source": "pubmed_pmc"
                })
                
        if all_chunks:
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks from {pmcid}...")
            emb_service = EmbeddingService()
            embeddings = emb_service.get_embeddings(all_chunks)
            add_documents(all_chunks, embeddings, all_metadatas)
            logger.info(f"Successfully stored {len(all_chunks)} chunks for {pmcid} in ChromaDB.")
            
        return abstract_text
    except Exception as e:
        logger.exception(f"Error processing PMC article {pmcid}: {e}")
        return ""


async def search_pubmed(query: str, max_results: int = MAX_RESULTS) -> dict:
    """Search PubMed for biomedical literature.

    Args:
        query: Biomedical search query string.
        max_results: Maximum number of articles to return (default 5).

    Returns:
        Structured dict with keys: tool, query, count, articles.
        Each article has: pmid, title, source, pub_date, authors, url, pmcid, abstract.
    """
    if not query or not query.strip():
        raise ValueError("query cannot be empty.")
    query = query.strip()
    if len(query) > 300:
        raise ValueError("query exceeds 300-character limit.")
    max_results = max(1, min(max_results, MAX_RESULTS))

    async def _run(verify_ssl: bool) -> dict:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, verify=verify_ssl) as client:
            try:
                # Step 1 — Fetch matching PMIDs
                esearch = await _robust_get(
                    client,
                    f"{ENTREZ_BASE}/esearch.fcgi",
                    params={
                        "db": "pubmed",
                        "term": query,
                        "retmax": max_results,
                        "retmode": "json",
                    },
                )
                esearch.raise_for_status()
                pmids: list[str] = esearch.json().get("esearchresult", {}).get("idlist", [])

                if not pmids:
                    return {"tool": "search_pubmed", "query": query, "count": 0, "articles": []}

                # Step 2 — Fetch article summaries for those PMIDs
                esummary = await _robust_get(
                    client,
                    f"{ENTREZ_BASE}/esummary.fcgi",
                    params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
                )
                esummary.raise_for_status()
                summaries = esummary.json().get("result", {})

                articles = []
                for pmid in pmids:
                    entry = summaries.get(pmid)
                    if not entry or pmid == "uids":
                        continue
                    
                    # Resilience extraction of PMCID
                    pmcid = None
                    for aid in entry.get("articleids", []):
                        if aid.get("idtype") == "pmc":
                            pmcid = aid.get("value")
                        elif aid.get("idtype") == "pmcid" and not pmcid:
                            val = aid.get("value") or aid.get("id") or ""
                            if "PMC" in val:
                                match = re.search(r"PMC\d+", val)
                                if match:
                                    pmcid = match.group(0)

                    title = _sanitize(entry.get("title"), 300)
                    
                    from app.rag.vectorstore import is_pmid_stored, get_stored_abstract
                    is_cached = is_pmid_stored(pmid)

                    # Delay to avoid hitting NCBI 3 req/sec rate limit
                    await asyncio.sleep(0.4)

                    # Conditional Fetching abstract or deep-fetch PMC full text
                    abstract = ""
                    if is_cached:
                        logger.info(f"PMID {pmid} already exists in ChromaDB. Skipping duplicate storage.")
                        abstract = get_stored_abstract(pmid)
                        if not abstract:
                            if "Has Abstract" in entry.get("attributes", []) or pmcid:
                                abstract = await _fetch_abstract_if_available(client, pmid)
                    else:
                        if not pmcid:
                            if "Has Abstract" in entry.get("attributes", []):
                                abstract = await _fetch_abstract_if_available(client, pmid)
                            else:
                                logger.info(f"No abstract available for PMID {pmid} (skipping retrieval).")
                        else:
                            abstract = await _process_and_store_pmc(client, pmid, pmcid, title)


                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "source": _sanitize(entry.get("source"), 100),
                        "pub_date": _sanitize(entry.get("pubdate"), 50),
                        "authors": [
                            _sanitize(a.get("name"), 100)
                            for a in entry.get("authors", [])[:5]
                        ],
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "pmcid": pmcid,
                        "abstract": abstract,
                    })

                return {
                    "tool": "search_pubmed",
                    "query": query,
                    "count": len(articles),
                    "articles": articles,
                }

            except httpx.TimeoutException:
                logger.warning("PubMed search timed out.")
                return {"tool": "search_pubmed", "query": query, "error": "Request timed out.", "articles": []}

            except httpx.HTTPStatusError as e:
                logger.error(f"PubMed HTTP error: {e.response.status_code}")
                return {"tool": "search_pubmed", "query": query, "error": f"HTTP {e.response.status_code}", "articles": []}

            except Exception as e:
                if isinstance(e, httpx.RequestError):
                    raise
                logger.exception("Unexpected PubMed error.")
                return {"tool": "search_pubmed", "query": query, "error": "Unexpected error.", "articles": []}

    try:
        return await _run(verify_ssl=True)
    except Exception as e:
        if isinstance(e, httpx.ConnectError) and "certificate verify failed" in str(e):
            logger.warning("NCBI PubMed TLS verification failed. Retrying with verify=False...")
            return await _run(verify_ssl=False)
        raise
