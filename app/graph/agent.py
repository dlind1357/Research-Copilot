"""LangGraph agent graph for the research copilot.

Graph flow:
  START
    └─▶ safety_check
          ├─▶ [BLOCKED] → END  (returns rejection message)
          └─▶ [PASSED]  → route_query
                              ├─▶ execute_rag    ─▶ generate_answer → END
                              ├─▶ execute_pubmed ─▶ generate_answer → END
                              └─▶ execute_gene   ─▶ generate_answer → END
"""

from __future__ import annotations

import logging
from langgraph.graph import StateGraph, END
from app.graph.state import AgentState
from app.graph.nodes import (
    safety_check,
    route_query,
    execute_rag,
    execute_pubmed,
    execute_gene,
    generate_answer,
)

logger = logging.getLogger(__name__)


# ── Conditional edges ─────────────────────────────────────────────────────────

def _after_safety(state: AgentState) -> str:
    """Route after safety check: pass → route_query, fail → END."""
    return "route_query" if state.get("safety_passed") else END


def _after_routing(state: AgentState) -> str:
    """Route to the appropriate tool executor based on tool_choice."""
    choice = state.get("tool_choice", "rag")
    return {
        "rag": "execute_rag",
        "pubmed": "execute_pubmed",
        "gene": "execute_gene",
    }.get(choice, "execute_rag")


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph research agent."""
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("safety_check", safety_check)
    builder.add_node("route_query", route_query)
    builder.add_node("execute_rag", execute_rag)
    builder.add_node("execute_pubmed", execute_pubmed)
    builder.add_node("execute_gene", execute_gene)
    builder.add_node("generate_answer", generate_answer)

    # Entry point
    builder.set_entry_point("safety_check")

    # Conditional edge: after safety check
    builder.add_conditional_edges("safety_check", _after_safety)

    # Conditional edge: after routing
    builder.add_conditional_edges("route_query", _after_routing)

    # All tool nodes flow into generate_answer
    builder.add_edge("execute_rag", "generate_answer")
    builder.add_edge("execute_pubmed", "generate_answer")
    builder.add_edge("execute_gene", "generate_answer")

    # generate_answer always terminates
    builder.add_edge("generate_answer", END)

    return builder.compile()


# ── Public API ────────────────────────────────────────────────────────────────

# Compiled graph singleton — import this for use in FastAPI
research_agent = build_graph()


async def run_agent(query: str) -> dict:
    """Run the full research agent pipeline for a given query.

    Args:
        query: The user's natural language research question.

    Returns:
        Dict with keys: final_answer, citations, tool_choice,
                        safety_passed, safety_reason, error.
    """
    initial_state: AgentState = {
        "query": query,
        "safety_passed": False,
        "safety_reason": None,
        "tool_choice": "none",
        "rag_results": [],
        "pubmed_results": [],
        "gene_result": None,
        "final_answer": "",
        "citations": [],
        "error": None,
    }

    try:
        final_state = await research_agent.ainvoke(initial_state)
    except Exception:
        logger.error("Agent graph execution failed.")
        return {
            "final_answer": "The research agent encountered an internal error. Please try again.",
            "citations": [],
            "tool_choice": "none",
            "safety_passed": False,
            "safety_reason": None,
            "error": "Agent execution failed.",
        }

    return {
        "final_answer": final_state.get("final_answer", ""),
        "citations": final_state.get("citations", []),
        "tool_choice": final_state.get("tool_choice", "none"),
        "safety_passed": final_state.get("safety_passed", False),
        "safety_reason": final_state.get("safety_reason"),
        "error": final_state.get("error"),
    }
