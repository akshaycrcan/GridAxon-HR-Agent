"""LangGraph policy routing with a bounded broad-search fallback; no answer LLM yet."""

from datetime import date
import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class RetrievalState(TypedDict, total=False):
    question: str
    methods: list[str]
    top_k: int
    as_of: str
    query_vector: list[float]
    route: dict
    results: dict
    broaden_methods: list[str]
    trace: list[str]
    warnings: list[str]


def choose_route(question, ranked, eligible, known_ids, min_score=0.35, margin=0.06):
    """Catalog-only heuristic. Thresholds are initial settings, not calibrated confidence."""
    explicit = sorted(set(re.findall(r"\bHR-\d{3}\b", question.upper())))
    if explicit:
        unknown = set(explicit) - set(known_ids)
        if unknown:
            raise ValueError(f"Unknown policy references: {', '.join(sorted(unknown))}")
        return {"mode": "explicit", "document_ids": [i for i in explicit if i in eligible],
                "reason": "Policy IDs explicitly named in the question", "candidates": ranked[:5]}
    if not ranked:
        return {"mode": "broad", "document_ids": [], "reason": "No eligible policies", "candidates": []}
    # Select up to three policies; do not force a single-document interpretation.
    if ranked[0]["score"] < min_score or (
            len(ranked) > 3 and ranked[0]["score"] - ranked[3]["score"] < margin):
        return {"mode": "broad", "document_ids": list(eligible),
                "reason": "Weak or ambiguous catalog ranking; search all eligible policies",
                "candidates": ranked[:5]}
    return {"mode": "routed", "document_ids": [item["document_id"] for item in ranked[:3]],
            "reason": "Top three policy titles/section catalogs by embedding similarity",
            "candidates": ranked[:5]}


def build_retrieval_graph(retriever, min_route_score=0.35, route_margin=0.06,
                          min_match_score=0.35):
    def embed_question(state):
        methods = state.get("methods", ["fixed", "semantic"])
        if not methods or any(method not in ("fixed", "semantic") for method in methods):
            raise ValueError("Select fixed, semantic, or both methods")
        top_k = state.get("top_k", 5)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 50:
            raise ValueError("top-k must be between 1 and 50")
        as_of = state.get("as_of") or date.today().isoformat()
        date.fromisoformat(as_of)
        return {"query_vector": retriever.embed_question(state["question"]),
                "methods": methods, "top_k": top_k, "as_of": as_of,
                "trace": ["embed_question"], "warnings": [
                    "Retrieval candidates only: relevance and answer sufficiency are not yet graded.",
                    "This index contains current policy versions only; historical coverage is incomplete.",
                ]}

    def route_documents(state):
        eligible = retriever.eligible_documents(state["as_of"])
        ranked = retriever.catalog_scores(state["query_vector"], eligible)
        route = choose_route(state["question"], ranked, eligible, retriever.catalog,
                             min_route_score, route_margin)
        return {"route": route, "trace": state["trace"] + ["route_documents"]}

    def retrieve(state):
        results, broaden = {}, []
        for method in state["methods"]:
            matches = retriever.query(state["query_vector"], method, state["top_k"],
                                      state["route"]["document_ids"], state["as_of"])
            results[method] = matches
            if state["route"]["mode"] == "routed" and (
                    len(matches) < state["top_k"] or not matches or matches[0]["score"] < min_match_score):
                broaden.append(method)
        return {"results": results, "broaden_methods": broaden,
                "trace": state["trace"] + ["retrieve"]}

    def broaden_search(state):
        results = dict(state["results"])
        for method in state["broaden_methods"]:
            results[method] = retriever.query(state["query_vector"], method, state["top_k"],
                                              as_of=state["as_of"])
        return {"results": results, "trace": state["trace"] + ["broaden_search"],
                "warnings": state["warnings"] + [
                    "A restricted search returned too few/weak candidates; used one broader search."]}

    graph = StateGraph(RetrievalState)
    graph.add_node("embed_question", embed_question)
    graph.add_node("route_documents", route_documents)
    graph.add_node("retrieve", retrieve)
    graph.add_node("broaden_search", broaden_search)
    graph.add_edge(START, "embed_question")
    graph.add_edge("embed_question", "route_documents")
    graph.add_edge("route_documents", "retrieve")
    graph.add_conditional_edges("retrieve", lambda state: "broaden" if state["broaden_methods"] else "done",
                                {"broaden": "broaden_search", "done": END})
    graph.add_edge("broaden_search", END)
    return graph.compile()


def run_routed(retriever, question, methods=("fixed", "semantic"), top_k=5, as_of=None):
    result = build_retrieval_graph(retriever).invoke(
        {"question": question, "methods": list(methods), "top_k": top_k, "as_of": as_of},
        config={"recursion_limit": 8},
    )
    result.pop("query_vector", None)
    return result
