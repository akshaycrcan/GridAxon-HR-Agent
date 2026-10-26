from copy import deepcopy
import unittest
from unittest.mock import patch

from gridaxon_hr_qa.grading import EvidenceGrader, GradingError, validate_grade, validate_plan, resolve_evidence_spans
from gridaxon_hr_qa.graded_workflow import run_graded


PLAN = {"requirements": [{"id": "R1", "description": "Annual PTO accrual"}],
        "clarification_question": ""}
MATCH = {"chunk_id": "c1", "text": "Eligible employees accrue 120 hours per year.",
         "citation": "[HR-008 v2.0 section 1]", "metadata": {"document_id": "HR-008"}}


def supported():
    return {"relevance": [{"chunk_id": "c1", "relevant": True, "reason": "States the accrual."}],
            "coverage": [{"requirement_id": "R1", "status": "supported", "reason": "Explicit rule.",
                          "evidence": [{"chunk_id": "c1", "quote": "120 hours per year"}]}],
            "conflicts": [], "clarification_question": ""}


def span_supported():
    raw = supported()
    raw["coverage"][0]["evidence"] = [{"chunk_id": "c1", "span_id": "S1"}]
    return raw


class Backend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete_json(self, system, payload):
        self.calls.append(payload)
        return deepcopy(self.replies.pop(0))


class GradingTests(unittest.TestCase):
    def test_supported_exact_evidence_opens_gate(self):
        result = validate_grade(supported(), PLAN, [MATCH])
        self.assertTrue(result["can_generate"])
        self.assertEqual(result["relevant_count"], 1)
        self.assertEqual(result["evidence_chunk_ids"], ["c1"])

    def test_partial_coverage_cannot_be_overridden_by_model(self):
        raw = supported()
        raw["coverage"][0]["status"] = "partial"
        raw["can_generate"] = True
        result = validate_grade(raw, PLAN, [MATCH])
        self.assertFalse(result["can_generate"])
        self.assertEqual(result["decision"], "retrieve_more")

    def test_many_relevant_chunks_do_not_replace_coverage(self):
        plan = deepcopy(PLAN)
        plan["requirements"].append({"id": "R2", "description": "Carryover limit"})
        raw = supported()
        raw["coverage"].append({"requirement_id": "R2", "status": "missing",
                                "reason": "No carryover rule provided.", "evidence": []})
        result = validate_grade(raw, plan, [MATCH])
        self.assertFalse(result["can_generate"])
        self.assertEqual(len(result["missing_requirements"]), 1)

    def test_invented_quote_or_chunk_id_rejected(self):
        for key, value in [("quote", "200 hours per year"), ("chunk_id", "invented")]:
            raw = supported()
            raw["coverage"][0]["evidence"][0][key] = value
            with self.assertRaises(GradingError):
                validate_grade(raw, PLAN, [MATCH])

    def test_omitted_requirement_or_relevance_rejected(self):
        for key in ("coverage", "relevance"):
            raw = supported()
            raw[key] = []
            with self.assertRaises(GradingError):
                validate_grade(raw, PLAN, [MATCH])

    def test_irrelevant_chunk_cannot_support_coverage(self):
        raw = supported()
        raw["relevance"][0]["relevant"] = False
        with self.assertRaises(GradingError):
            validate_grade(raw, PLAN, [MATCH])

    def test_conflict_and_clarification_block_generation(self):
        raw = supported()
        raw["conflicts"] = [{"description": "Conflicting dates in passage", "chunk_ids": ["c1"]}]
        self.assertEqual(validate_grade(raw, PLAN, [MATCH])["decision"], "resolve_conflict")
        raw = supported()
        raw["clarification_question"] = "Which employment category applies?"
        self.assertEqual(validate_grade(raw, PLAN, [MATCH])["decision"], "clarify")

    def test_empty_results_need_no_grading_api_call(self):
        backend = Backend([])
        result = EvidenceGrader(backend).grade("PTO?", PLAN, [], "2026-10-25")
        self.assertEqual(result["decision"], "retrieve_more")
        self.assertFalse(backend.calls)

    def test_plan_is_independent_of_retrieved_evidence(self):
        backend = Backend([PLAN, span_supported()])
        grader = EvidenceGrader(backend)
        plan = grader.plan("How much PTO?")
        result = grader.grade("How much PTO?", plan, [MATCH], "2026-10-25")
        self.assertEqual(backend.calls[0], {"question": "How much PTO?"})
        self.assertNotIn("score", backend.calls[1]["passages"][0])
        self.assertTrue(result["can_generate"])

    def test_invalid_plan_rejected(self):
        for raw in ({}, {"requirements": [], "clarification_question": ""},
                    {"requirements": PLAN["requirements"] * 2, "clarification_question": ""}):
            with self.assertRaises(GradingError):
                validate_plan(raw)


class WorkflowTests(unittest.TestCase):
    def retrieval(self, mode="routed"):
        return {"results": {"fixed": [MATCH], "semantic": []}, "as_of": "2026-10-25",
                "top_k": 5, "route": {"mode": mode, "document_ids": ["HR-008"]}}

    def test_retry_one_arm_and_gate_arms_independently(self):
        class Retriever:
            calls = []
            def embed_question(self, question):
                return [1, 0]
            def query(self, vector, method, top_k, scope, as_of):
                self.calls.append((method, top_k, scope, as_of))
                return []
        retriever = Retriever()
        grader = EvidenceGrader(Backend([PLAN, span_supported()]))
        with patch("gridaxon_hr_qa.graded_workflow.run_routed", return_value=self.retrieval()):
            result = run_graded(retriever, grader, "PTO?")
        self.assertEqual(result["generation_ready"], {"fixed": True, "semantic": False})
        self.assertEqual(result["grades"]["semantic"]["decision"], "abstain")
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(retriever.calls[0], ("semantic", 10, None, "2026-10-25"))

    def test_retry_preserves_explicit_scope_and_regrades_new_evidence(self):
        class Retriever:
            scope = None
            def embed_question(self, question):
                return [1, 0]
            def query(self, vector, method, top_k, scope, as_of):
                self.scope = scope
                return [MATCH]
        retriever = Retriever()
        grader = EvidenceGrader(Backend([PLAN, span_supported(), span_supported()]))
        with patch("gridaxon_hr_qa.graded_workflow.run_routed", return_value=self.retrieval("explicit")):
            result = run_graded(retriever, grader, "HR-008 PTO?")
        self.assertEqual(retriever.scope, ["HR-008"])
        self.assertTrue(all(result["generation_ready"].values()))

    def test_invalid_grade_fails_closed(self):
        raw = span_supported()
        raw["coverage"][0]["evidence"][0]["span_id"] = "invented"
        grader = EvidenceGrader(Backend([PLAN, raw]))
        with patch("gridaxon_hr_qa.graded_workflow.run_routed", return_value=self.retrieval()):
            with self.assertRaises(GradingError):
                run_graded(object(), grader, "PTO?")


class SourceSpanTests(unittest.TestCase):
    def test_exact_source_text_is_copied_with_unicode_and_spacing(self):
        match = {**MATCH, "text": "## PTO\r\nEligible employees  accrue 120 hours — annually.\r\n"}
        raw = span_supported()
        raw["coverage"][0]["evidence"][0]["span_id"] = "S2"
        resolved = resolve_evidence_spans(raw, [match])
        citation = resolved["coverage"][0]["evidence"][0]
        self.assertEqual(citation["quote"], "Eligible employees  accrue 120 hours — annually.")
        self.assertEqual(citation["quote"], match["text"][citation["quote_start_char"]:citation["quote_end_char"]])
        self.assertTrue(validate_grade(resolved, PLAN, [match])["can_generate"])
        self.assertNotIn("quote", raw["coverage"][0]["evidence"][0])

    def test_wrong_chunk_cannot_borrow_another_chunks_span(self):
        matches = [MATCH, {**MATCH, "chunk_id": "c2", "text": "First line\nSecond line"}]
        raw = span_supported()
        raw["coverage"][0]["evidence"][0]["span_id"] = "S2"
        with self.assertRaises(GradingError):
            resolve_evidence_spans(raw, matches)

    def test_generated_quote_alone_is_not_accepted_from_backend(self):
        with self.assertRaises(GradingError):
            resolve_evidence_spans(supported(), [MATCH])
