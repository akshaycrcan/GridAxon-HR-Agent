"""A generation gate after retrieval, with at most one evidence-retrieval retry."""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from gridaxon_hr_qa.routing import run_routed


class GradedState(TypedDict, total=False):
    question: str
    methods: list[str]
    top_k: int
    as_of: str
    retrieval: dict
    plan: dict
    grades: dict
    first_grades: dict
    retry_methods: list[str]
    trace: list[str]
    generation_ready: dict


def build_graded_graph(retriever, grader):
    def plan_question(state):
        return {"plan": grader.plan(state["question"]), "trace": ["plan_question"]}

    def retrieve(state):
        result = run_routed(retriever, state["question"], state.get("methods", ["fixed", "semantic"]),
                            state.get("top_k", 5), state.get("as_of"))
        return {"retrieval": result, "trace": state["trace"] + ["retrieve_and_route"]}

    def grade(state):
        retrieval = state["retrieval"]
        grades = {method: grader.grade(state["question"], state["plan"], matches, retrieval["as_of"])
                  for method, matches in retrieval["results"].items()}
        retry = [method for method, result in grades.items() if result["decision"] == "retrieve_more"]
        return {"grades": grades, "first_grades": grades, "retry_methods": retry,
                "trace": state["trace"] + ["grade_evidence"]}

    def retry_retrieval(state):
        previous = state["retrieval"]
        results = dict(previous["results"])
        grades = dict(state["grades"])
        vector = retriever.embed_question(state["question"])
        scope = previous["route"]["document_ids"] if previous["route"]["mode"] == "explicit" else None
        wider_k = min(50, max(previous["top_k"] * 2, 10))
        for method in state["retry_methods"]:
            matches = retriever.query(vector, method, wider_k, scope, previous["as_of"])
            # Never reuse an earlier grade after the evidence set changes.
            if {item["chunk_id"] for item in matches} != {item["chunk_id"] for item in results[method]}:
                results[method] = matches
                grades[method] = grader.grade(state["question"], state["plan"], matches, previous["as_of"])
        return {"retrieval": {**previous, "results": results, "retry_top_k": wider_k},
                "grades": grades, "trace": state["trace"] + ["retry_retrieval_and_grade"]}

    def finalize(state):
        grades = {method: dict(result) for method, result in state["grades"].items()}
        for result in grades.values():
            if result["decision"] == "retrieve_more":
                result["decision"] = "abstain"
        return {"grades": grades, "generation_ready": {
                    method: result["can_generate"] for method, result in grades.items()},
                "trace": state["trace"] + ["generation_gate"]}

    graph = StateGraph(GradedState)
    graph.add_node("plan_question", plan_question)
    graph.add_node("retrieve_and_route", retrieve)
    graph.add_node("grade_evidence", grade)
    graph.add_node("retry_retrieval_and_grade", retry_retrieval)
    graph.add_node("generation_gate", finalize)
    graph.add_edge(START, "plan_question")
    graph.add_edge("plan_question", "retrieve_and_route")
    graph.add_edge("retrieve_and_route", "grade_evidence")
    graph.add_conditional_edges("grade_evidence", lambda state: "retry" if state["retry_methods"] else "finish",
                                {"retry": "retry_retrieval_and_grade", "finish": "generation_gate"})
    graph.add_edge("retry_retrieval_and_grade", "generation_gate")
    graph.add_edge("generation_gate", END)
    return graph.compile()


def run_graded(retriever, grader, question, methods=("fixed", "semantic"), top_k=5, as_of=None):
    return build_graded_graph(retriever, grader).invoke(
        {"question": question, "methods": list(methods), "top_k": top_k, "as_of": as_of},
        config={"recursion_limit": 8},
    )
