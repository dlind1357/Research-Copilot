"""Gene information lookup tool using MyGene.info API."""

import re
import logging
import httpx

logger = logging.getLogger(__name__)

MYGENE_BASE = "https://mygene.info/v3"
MAX_FIELD_LENGTH = 500
TIMEOUT_SECONDS = 5.0


def _sanitize(text: str | None, max_len: int = MAX_FIELD_LENGTH) -> str:
    """Strip control characters and truncate."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", str(text))
    return (text[:max_len] + "…") if len(text) > max_len else text.strip()


async def lookup_gene(gene: str) -> dict:
    """Look up human gene information by symbol or name via MyGene.info.

    Args:
        gene: Gene symbol or name (e.g. "BRCA1", "TP53", "EGFR").

    Returns:
        Structured dict with keys: tool, gene, found, result.
        result contains: symbol, name, summary, aliases,
                         chromosome, start, end, mim_id.

    Raises:
        ValueError: If gene is empty, too long, or contains invalid characters.
    """
    if not gene or not gene.strip():
        raise ValueError("gene cannot be empty.")
    gene = gene.strip()
    if len(gene) > 100:
        raise ValueError("gene exceeds 100-character limit.")
    if not re.match(r"^[A-Za-z0-9\-]+$", gene):
        raise ValueError("gene must contain only letters, digits, and hyphens.")

    async def _run(verify_ssl: bool) -> dict:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, verify=verify_ssl) as client:
            try:
                resp = await client.get(
                    f"{MYGENE_BASE}/query",
                    params={
                        "q": gene,
                        "fields": "symbol,name,summary,alias,genomic_pos,MIM",
                        "species": "human",
                        "size": 1,
                    },
                )
                resp.raise_for_status()
                hits = resp.json().get("hits", [])

                if not hits:
                    return {"tool": "lookup_gene", "gene": gene, "found": False, "result": None}

                hit = hits[0]

                # genomic_pos can be a dict or list depending on the gene
                genomic_pos = hit.get("genomic_pos", {})
                if isinstance(genomic_pos, list):
                    genomic_pos = genomic_pos[0] if genomic_pos else {}

                aliases = hit.get("alias", []) or []
                if isinstance(aliases, str):
                    aliases = [aliases]

                result = {
                    "symbol": _sanitize(hit.get("symbol"), 50),
                    "name": _sanitize(hit.get("name"), 200),
                    "summary": _sanitize(hit.get("summary"), MAX_FIELD_LENGTH),
                    "aliases": [_sanitize(a, 50) for a in aliases[:10]],
                    "chromosome": _sanitize(str(genomic_pos.get("chr", "")), 10),
                    "start": genomic_pos.get("start"),
                    "end": genomic_pos.get("end"),
                    "mim_id": hit.get("MIM"),
                    "ncbi_gene_id": _sanitize(str(hit.get("_id", "")), 20),
                }

                return {"tool": "lookup_gene", "gene": gene, "found": True, "result": result}

            except httpx.TimeoutException:
                logger.warning(f"Gene lookup timed out for: {gene}")
                return {"tool": "lookup_gene", "gene": gene, "error": "Request timed out after 5s.", "found": False, "result": None}

            except httpx.HTTPStatusError as e:
                logger.error(f"Gene lookup HTTP error: {e.response.status_code}")
                return {"tool": "lookup_gene", "gene": gene, "error": f"HTTP {e.response.status_code}", "found": False, "result": None}

            except Exception as e:
                if isinstance(e, httpx.RequestError):
                    raise
                logger.exception("Unexpected gene lookup error.")
                return {"tool": "lookup_gene", "gene": gene, "found": False, "result": None}

    try:
        return await _run(verify_ssl=True)
    except Exception as e:
        if isinstance(e, httpx.ConnectError) and "certificate verify failed" in str(e):
            logger.warning("MyGene TLS verification failed. Retrying with verify=False...")
            return await _run(verify_ssl=False)
        raise
