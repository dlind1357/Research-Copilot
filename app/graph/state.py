"""Shared state schema for the LangGraph research agent."""

from typing import Annotated, Literal
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    """Immutable state passed between graph nodes."""

    # ── Input ─────────────────────────────────────────────────────────────────
    query: str                           # Original user query

    # ── Safety ────────────────────────────────────────────────────────────────
    safety_passed: bool                  # True if query passed safety check
    safety_reason: str | None           # Human-readable rejection reason

    # ── Routing ───────────────────────────────────────────────────────────────
    tool_choice: Literal["rag", "pubmed", "gene", "none"]

    # ── Tool results (treated as untrusted external data) ─────────────────────
    rag_results: list[dict]
    pubmed_results: list[dict]
    gene_result: dict | None

    # ── Output ────────────────────────────────────────────────────────────────
    final_answer: str
    citations: Annotated[list[dict], operator.add]  # Accumulate citations
    error: str | None
