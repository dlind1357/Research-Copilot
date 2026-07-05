from fastapi import FastAPI, UploadFile, File, HTTPException
from app.api.models import ChatRequest, ChatResponse, Citation, EvaluationRequest, EvaluationResponse
from app.tools.gcs import upload_file
from app.rag.loader import extract_text_from_pdf
from app.rag.chunker import chunk_text
from app.rag.retriever import retrieve_context
from app.mcp.server import search_pubmed, lookup_gene, search_rag
from app.graph.agent import run_agent
from app.security.guardrails import filter_untrusted_document
from app.eval.evaluator import evaluate_response
import os
import tempfile
import uuid

app = FastAPI(
    title="Research Copilot API",
    description="A production-ready Python FastAPI project using RAG over scientific papers.",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {"message": "Welcome to Research Copilot API"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/health/live")
async def health_live():
    return {"status": "live"}

@app.get("/health/ready")
async def health_ready():
    return {"status": "ready"}

@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check validating internal services and external APIs."""
    import asyncio
    import httpx
    from app.rag.vectorstore import count_documents
    from app.rag.embeddings import EmbeddingService
    from app.config.settings import settings

    # 1. API Status (Simple app responsiveness)
    api_status = {"status": "healthy"}

    # 2. RAG Status (Verify local ChromaDB accessibility)
    try:
        doc_count = count_documents()
        rag_status = {
            "status": "healthy",
            "details": {
                "document_count": doc_count,
                "collection": "research_papers"
            }
        }
    except Exception as e:
        rag_status = {"status": "unhealthy", "error": str(e)}

    # 3. Embedding Service Status (Verify local embedding creation with timeout)
    try:
        service = EmbeddingService()
        loop = asyncio.get_running_loop()
        # Execute the synchronous get_embedding in a background thread to prevent blocking
        test_emb = await asyncio.wait_for(
            loop.run_in_executor(None, service.get_embedding, "healthcheck"),
            timeout=3.0
        )
        embedding_status = {
            "status": "healthy",
            "details": {
                "model": service.model_name,
                "dimension": len(test_emb) if test_emb else 0
            }
        }
    except Exception as e:
        embedding_status = {
            "status": "unhealthy",
            "error": f"Failed or timed out during generation: {str(e)}"
        }

    # Check external integrations concurrently to minimize latency
    async with httpx.AsyncClient(timeout=4.0) as client:
        
        async def check_mcp_pubmed():
            try:
                # E-utilities fast endpoint check
                r = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params={"db": "pubmed", "term": "health", "retmax": 1, "retmode": "json"}
                )
                if r.status_code == 200:
                    return {"status": "healthy", "details": "NCBI E-utilities API is reachable"}
                else:
                    return {"status": "unhealthy", "error": f"HTTP {r.status_code}"}
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}

        async def check_mcp_gene():
            try:
                # MyGene query endpoint check
                r = await client.get("https://mygene.info/v3/query", params={"q": "TP53", "size": 1})
                if r.status_code == 200:
                    return {"status": "healthy", "details": "MyGene.info API is reachable"}
                else:
                    return {"status": "unhealthy", "error": f"HTTP {r.status_code}"}
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}

        async def check_external_api():
            try:
                # Direct check of Gemini Model Directory API using our key
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={settings.GOOGLE_API_KEY}"
                r = await client.get(url)
                if r.status_code == 200:
                    return {
                        "status": "healthy",
                        "details": "Google Generative AI service is reachable and key is valid."
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"Gemini API returned status code {r.status_code}"
                    }
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}

        pubmed_res, gene_res, ext_res = await asyncio.gather(
            check_mcp_pubmed(),
            check_mcp_gene(),
            check_external_api()
        )

    # 4. MCP Status aggregates external research lookup service statuses
    mcp_healthy = (pubmed_res["status"] == "healthy" and gene_res["status"] == "healthy")
    mcp_status = {
        "status": "healthy" if mcp_healthy else "unhealthy",
        "pubmed": pubmed_res,
        "mygene": gene_res
    }

    # 5. External API Status is verified by the Gemini API connectivity check
    external_api_status = ext_res

    # Overall system health rollup
    overall_healthy = (
        api_status["status"] == "healthy" and
        rag_status["status"] == "healthy" and
        embedding_status["status"] == "healthy" and
        mcp_status["status"] == "healthy" and
        external_api_status["status"] == "healthy"
    )

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "components": {
            "api": api_status,
            "rag": rag_status,
            "mcp": mcp_status,
            "embedding_service": embedding_status,
            "external_api": external_api_status
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Run the full LangGraph research agent pipeline."""
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in request.")

    query = user_messages[-1].content

    result = await run_agent(query)

    # Safety rejection
    if not result.get("safety_passed"):
        return ChatResponse(
            response=f"Request blocked: {result.get('safety_reason', 'Query not allowed.')}",
            citations=[],
            context_used=False,
            num_chunks_retrieved=0,
        )

    citations = [
        Citation(
            chunk_id=c.get("chunk_id") or c.get("pmid") or c.get("symbol") or "unknown",
            text=c.get("text") or c.get("title") or c.get("summary") or "",
            distance=c.get("distance", 0.0),
            metadata={k: v for k, v in c.items() if k not in {"text", "distance", "chunk_id"}},
        )
        for c in result.get("citations", [])
    ]

    return ChatResponse(
        response=result.get("final_answer", ""),
        citations=citations,
        context_used=bool(citations),
        num_chunks_retrieved=len(citations),
    )

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
BUCKET_NAME = "bio-copilot-data"

@app.post("/upload-paper")
async def upload_paper(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the 10MB limit.")

    # Save PDF based on selected storage (local directory or GCS path)
    pdf_blob_name = f"{uuid.uuid4()}_{file.filename}"
    from app.tools.gcs import save_paper
    save_paper(content, pdf_blob_name)

    # Write to a temp file so PyMuPDF can read it via file_path
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        extracted_text = extract_text_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)  # Always clean up the temp file

    # Filter/sanitize untrusted document content for security guardrails
    extracted_text = filter_untrusted_document(extracted_text)

    # Chunk the extracted text
    chunks = chunk_text(extracted_text)

    return {
        "message": "File uploaded and processed successfully.",
        "pdf_filename": pdf_blob_name,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }


# ── MCP Tool endpoints ──────────────────────────────────────────────────────

@app.get("/tools/pubmed")
async def tool_pubmed(query: str):
    """Search PubMed for biomedical literature."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query parameter is required.")
    try:
        return await search_pubmed(query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tools/gene")
async def tool_gene(gene: str):
    """Look up gene information by symbol (e.g. BRCA1, TP53)."""
    if not gene or not gene.strip():
        raise HTTPException(status_code=400, detail="gene parameter is required.")
    try:
        return await lookup_gene(gene)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tools/rag")
async def tool_rag(query: str):
    """Search the local RAG vector store for relevant research chunks."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query parameter is required.")
    try:
        return await search_rag(query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """Run full RAG pipeline evaluation metrics."""
    try:
        results = evaluate_response(
            query=request.query,
            answer=request.answer,
            contexts=request.contexts,
            citations=request.citations,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
