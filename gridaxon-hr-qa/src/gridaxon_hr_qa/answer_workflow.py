"""LangGraph answer generation, verification, one revision, and publication gate."""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from gridaxon_hr_qa.graded_workflow import run_graded
from gridaxon_hr_qa.grading import GradingError
from gridaxon_hr_qa.answering import approved_sources, validate_draft, render_answer


class AnswerState(TypedDict, total=False):
    question: str
    methods: list[str]
    top_k: int
    as_of: str
    graded: dict
    sources: dict
    drafts: dict
    checks: dict
    attempts: dict
    retry_methods: list[str]
    answers: dict
    trace: list[str]


def build_answer_graph(retriever, grader, engine):
    def assess(state):
        graded = run_graded(retriever, grader, state["question"], state.get("methods", ["fixed", "semantic"]),
                            state.get("top_k", 5), state.get("as_of"))
        sources = {method: approved_sources(grade) for method, grade in graded["grades"].items()
                   if grade["can_generate"]}
        return {"graded": graded, "sources": sources, "attempts": {}, "trace": ["grade_evidence"]}

    def generate(state):
        drafts = {method: engine.generate(state["question"], state["graded"]["plan"], sources)
                  for method, sources in state["sources"].items()}
        return {"drafts": drafts, "trace": state["trace"] + ["generate_answers"]}

    def check(state, methods):
        checks, attempts = dict(state.get("checks", {})), dict(state["attempts"])
        for method in methods:
            raw = state["drafts"][method]
            try:
                draft = validate_draft(raw, state["sources"][method])
                result = engine.verify(state["question"], state["graded"]["plan"], draft, state["sources"][method])
            except GradingError as exc:
                result = {"passed": False, "issues": [str(exc)], "validation_error": True}
            checks[method] = result
            attempts[method] = attempts.get(method, []) + [{"draft": raw, "check": result}]
        return {"checks": checks, "attempts": attempts,
                "retry_methods": [method for method in methods if not checks[method]["passed"]]}

    def verify(state):
        return {**check(state, state["drafts"]), "trace": state["trace"] + ["check_groundedness_and_citations"]}

    def revise(state):
        drafts = dict(state["drafts"])
        for method in state["retry_methods"]:
            feedback = [{"previous_draft": drafts[method], "issues": state["checks"][method]["issues"]}]
            drafts[method] = engine.generate(state["question"], state["graded"]["plan"],
                                             state["sources"][method], feedback)
        return {"drafts": drafts, "trace": state["trace"] + ["revise_once"]}

    def verify_revision(state):
        return {**check(state, state["retry_methods"]), "trace": state["trace"] + ["recheck_revision"]}

    def publish(state):
        answers = {}
        for method, grade in state["graded"]["grades"].items():
            if not grade["can_generate"]:
                message = (grade["clarification_question"] if grade["decision"] == "clarify" else
                           "The retrieved evidence conflicts; this question needs review." if grade["decision"] == "resolve_conflict"
                           else "The retrieved policies do not provide enough evidence to answer this question.")
                answers[method] = {"status": grade["decision"], "validated": False, "text": message, "sources": []}
            elif state["checks"][method]["passed"]:
                answers[method] = {"status": "answered", "validated": True,
                                   **render_answer(validate_draft(state["drafts"][method], state["sources"][method]),
                                                   state["sources"][method])}
            else:
                answers[method] = {"status": "abstain", "validated": False, "sources": [],
                                   "text": "I could not validate a fully supported answer against the retrieved evidence."}
        return {"answers": answers, "trace": state["trace"] + ["publish_validated_answers"]}

    graph = StateGraph(AnswerState)
    for name, node in [("grade_evidence", assess), ("generate_answers", generate),
                       ("check_groundedness_and_citations", verify), ("revise_once", revise),
                       ("recheck_revision", verify_revision), ("publish_validated_answers", publish)]:
        graph.add_node(name, node)
    graph.add_edge(START, "grade_evidence")
    graph.add_edge("grade_evidence", "generate_answers")
    graph.add_edge("generate_answers", "check_groundedness_and_citations")
    graph.add_conditional_edges("check_groundedness_and_citations",
                                lambda state: "revise" if state["retry_methods"] else "publish",
                                {"revise": "revise_once", "publish": "publish_validated_answers"})
    graph.add_edge("revise_once", "recheck_revision")
    graph.add_edge("recheck_revision", "publish_validated_answers")
    graph.add_edge("publish_validated_answers", END)
    return graph.compile()


def run_answered(retriever, grader, engine, question, methods=("fixed", "semantic"), top_k=5, as_of=None):
    return build_answer_graph(retriever, grader, engine).invoke(
        {"question": question, "methods": list(methods), "top_k": top_k, "as_of": as_of},
        config={"recursion_limit": 10},
    )
