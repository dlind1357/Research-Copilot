"""Evaluation module for assessing RAG pipeline metrics (relevance, hallucination risk, completeness, and citation validation)."""

import json
import logging
import re
import httpx
import urllib3
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Suppress insecure request warnings for fallback SSL bypass
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


from app.config.llm import generate_llm_content

def _generate_via_rest(prompt: str, response_mime_type: str = "application/json") -> str:
    """Generate content using Vertex AI / Studio fallback generator helper."""
    return generate_llm_content(prompt, response_mime_type=response_mime_type)


def _run_llm_evaluation(prompt: str, default_score: float, heuristic_fn) -> tuple[float, str]:
    """Helper to query Gemini REST API with a fallback to programmatic heuristics if downstream APIs fail."""
    try:
        text_response = _generate_via_rest(prompt)
        data = json.loads(text_response.strip())
        score = float(data.get("score", default_score))
        reason = str(data.get("reason", "No reason provided."))
        score = max(0.0, min(1.0, score))
        return score, reason
    except Exception as e:
        logger.warning(f"LLM-assisted evaluation failed: {e}. Falling back to programmatic heuristic scoring.")
        # Execute the heuristic callback to avoid crashing
        return heuristic_fn()


def _heuristic_relevance(query: str, answer: str) -> tuple[float, str]:
    """Heuristic fallback for relevance based on term overlap."""
    stop_words = {"does", "do", "play", "a", "role", "in", "what", "is", "chromosome", "located", "on"}
    q_words = {w for w in re.findall(r"\w+", query.lower()) if w not in stop_words and len(w) > 2}
    a_words = {w for w in re.findall(r"\w+", answer.lower()) if w not in stop_words and len(w) > 2}
    if not q_words:
        return 1.0, "Query has no substantive terms."
    overlap = len(q_words & a_words) / len(q_words)
    # Scale from 0.5 to 1.0 based on overlap
    score = round(min(1.0, 0.5 + (overlap * 0.5)), 2)
    return score, f"Heuristic calculation: detected {round(overlap * 100)}% query term overlap with response."


def _heuristic_hallucination(answer: str, contexts: list[str]) -> tuple[float, str]:
    """Heuristic fallback for hallucination based on context entity overlap."""
    if not contexts:
        return 0.5, "No context provided to evaluate hallucination risk."
    
    stop_words = {
        "the", "a", "an", "and", "is", "of", "to", "in", "for", "on", "with", "that", "this", "are", "role", "play", "does",
        "yes", "according", "scientific", "research", "critical", "highly", "involved", "by", "from", "be", "been", "has", "have", "as"
    }
    a_words = [w for w in re.findall(r"\w+", answer.lower()) if w not in stop_words and len(w) > 2]
    if not a_words:
        return 0.0, "Heuristic: Answer contains no content words to evaluate."

    context_text = " ".join(contexts).lower()
    missing_count = sum(1 for w in a_words if w not in context_text)
    missing_ratio = missing_count / len(a_words)
    
    # Scale risk: highly missing implies high hallucination risk
    risk = round(min(1.0, missing_ratio), 2)
    return risk, f"Heuristic calculation: {round(missing_ratio * 100)}% of substantive answer terms were not found in contexts."


def _heuristic_completeness(query: str, answer: str) -> tuple[float, str]:
    """Heuristic fallback for completeness based on length and response presence."""
    if len(answer.strip()) < 20:
        return 0.2, "Heuristic: Response is extremely short."
    negatives = {"don't know", "do not know", "could not find", "insufficient", "no information"}
    if any(neg in answer.lower() for neg in negatives):
        return 0.4, "Heuristic: Response indicates insufficient context found."
    return 0.85, "Heuristic: Response of reasonable length generated."


def evaluate_relevance(query: str, answer: str) -> tuple[float, str]:
    """Calculate the relevance score of the answer to the user query."""
    prompt = f"""You are an objective AI evaluator. Rate the RELEVANCE of the following answer to the user's query.
The score must be a float between 0.0 (completely irrelevant or off-topic) and 1.0 (perfectly addresses the query).

Query: {query}
Answer: {answer}

Provide your evaluation as a JSON object matching this schema:
{{
  "score": float,
  "reason": "explanation of the rating"
}}"""
    return _run_llm_evaluation(prompt, default_score=0.5, heuristic_fn=lambda: _heuristic_relevance(query, answer))


def evaluate_hallucination_risk(answer: str, contexts: list[str]) -> tuple[float, str]:
    """Evaluate the hallucination risk of the answer based ONLY on the retrieved contexts."""
    context_block = "\n\n---\n\n".join(contexts) if contexts else "[No Context Provided]"
    prompt = f"""You are an objective AI evaluator. Rate the HALLUCINATION RISK of the answer below against the provided context.
- A score of 0.0 means the answer is 100% supported by the context with zero hallucination.
- A score of 1.0 means the answer is completely fabricated or directly contradicts the context.

Context:
{context_block}

Answer:
{answer}

Provide your evaluation as a JSON object matching this schema:
{{
  "score": float,
  "reason": "explanation of why the answer contains or does not contain hallucination"
}}"""
    return _run_llm_evaluation(prompt, default_score=1.0, heuristic_fn=lambda: _heuristic_hallucination(answer, contexts))


def evaluate_completeness(query: str, answer: str) -> tuple[float, str]:
    """Calculate the completeness score of how fully the answer addresses all parts of the query."""
    prompt = f"""You are an objective AI evaluator. Rate the COMPLETENESS of the answer below relative to the user's query.
The score must be a float between 0.0 (does not answer any part of the query) and 1.0 (fully and thoroughly answers all parts of the query).

Query: {query}
Answer: {answer}

Provide your evaluation as a JSON object matching this schema:
{{
  "score": float,
  "reason": "explanation of what parts are covered and what, if anything, is missing"
}}"""
    return _run_llm_evaluation(prompt, default_score=0.5, heuristic_fn=lambda: _heuristic_completeness(query, answer))


def validate_citations(contexts: list[dict], citations: list[dict]) -> tuple[float, list[dict]]:
    """Programmatically validate citations.
    
    Checks if citations point to existing items in contexts and verify text similarity/overlap.
    """
    if not citations:
        return 1.0, []

    validated = []
    valid_count = 0

    context_map = {}
    for ctx in contexts:
        text_content = ctx.get("text") or ctx.get("document") or ctx.get("summary") or ctx.get("title") or ""
        for key in ["id", "chunk_id", "pmid", "symbol"]:
            if key in ctx and ctx[key]:
                context_map[str(ctx[key])] = text_content

    for cit in citations:
        cid = str(cit.get("chunk_id") or cit.get("pmid") or cit.get("symbol") or "")
        cit_text = (cit.get("text") or "").strip()
        
        status = "invalid"
        reason = "Citation ID not found in retrieved contexts."
        
        if cid in context_map:
            ref_text = context_map[cid]
            if not cit_text:
                status = "valid"
                reason = "Citation ID matches context (no verification text provided)."
            elif cit_text.lower() in ref_text.lower():
                status = "valid"
                reason = "Citation text is an exact substring of the reference context."
            else:
                stop_words = {"the", "a", "an", "and", "is", "of", "to", "in", "for", "on", "with", "that", "this", "are", "be", "or", "as"}
                ref_words = {w for w in re.findall(r"\w+", ref_text.lower()) if w not in stop_words and len(w) > 2}
                cit_words = {w for w in re.findall(r"\w+", cit_text.lower()) if w not in stop_words and len(w) > 2}
                if not cit_words:
                    status = "suspicious"
                    reason = "Citation text contains only common stop words."
                else:
                    overlap = len(ref_words & cit_words) / len(cit_words)
                    if overlap >= 0.5:
                        status = "valid"
                        reason = f"Citation text matches reference context with {round(overlap * 100, 1)}% non-stop word overlap."
                    else:
                        status = "suspicious"
                        reason = f"Citation text has extremely low non-stop word overlap ({round(overlap * 100, 1)}%) with matching context."

        if status == "valid":
            valid_count += 1
            
        validated.append({
            "citation": cit,
            "status": status,
            "reason": reason
        })

    score = valid_count / len(citations) if citations else 1.0
    return score, validated


def evaluate_response(query: str, answer: str, contexts: list[dict], citations: list[dict]) -> dict:
    """Run full evaluation suite on a RAG query and response."""
    raw_contexts = [
        str(c.get("text") or c.get("document") or c.get("summary") or c.get("title") or "")
        for c in contexts
    ]

    rel_score, rel_reason = evaluate_relevance(query, answer)
    hall_score, hall_reason = evaluate_hallucination_risk(answer, raw_contexts)
    comp_score, comp_reason = evaluate_completeness(query, answer)
    cit_score, cit_details = validate_citations(contexts, citations)

    metrics = {
        "relevance_score": rel_score,
        "hallucination_risk": hall_score,
        "completeness_score": comp_score,
        "citation_validation_score": cit_score,
    }

    logger.info(f"EVALUATION METRICS: {metrics}")

    return {
        "metrics": metrics,
        "details": {
            "relevance": {"score": rel_score, "reason": rel_reason},
            "hallucination_risk": {"score": hall_score, "reason": hall_reason},
            "completeness": {"score": comp_score, "reason": comp_reason},
            "citations": {"score": cit_score, "validation": cit_details},
        }
    }
