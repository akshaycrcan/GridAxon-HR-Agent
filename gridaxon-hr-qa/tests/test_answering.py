from copy import deepcopy
import unittest
from unittest.mock import patch

from gridaxon_hr_qa.answering import approved_sources, validate_draft, validate_answer_check, AnswerEngine, render_answer
from gridaxon_hr_qa.answer_workflow import run_answered
from gridaxon_hr_qa.grading import GradingError


PLAN = {"requirements": [{"id": "R1", "description": "Annual accrual"}], "clarification_question": ""}
SOURCE = {"chunk_id": "c1", "span_id": "S1", "text": "Full-time staff accrue 120 hours annually.",
          "citation": "[HR-008 v2.0 section 1]", "metadata": {"title": "PTO"}}
DRAFT = {"claims": [{"claim_id": "A1", "text": "Full-time staff accrue 120 hours each year.",
                     "citations": [{"chunk_id": "c1", "span_id": "S1"}]}]}
CHECK = {"claim_checks": [{"claim_id": "A1", "supported": True, "reason": "Explicit accrual rule."}],
         "coverage": [{"requirement_id": "R1", "covered": True, "claim_ids": ["A1"], "reason": "Amount stated."}]}


def grade(ready=True):
    return {"can_generate": ready, "evidence_sufficient": ready, "decision": "ready" if ready else "abstain",
            "clarification_question": "", "coverage": [{"requirement_id": "R1", "status": "supported",
            "evidence": [{"chunk_id": "c1", "span_id": "S1", "quote": SOURCE["text"]}]}],
            "retained_matches": [{"chunk_id": "c1", "text": SOURCE["text"],
                                   "citation": SOURCE["citation"], "metadata": {"title": "PTO", "text": "uncited secret"}}]}


class Backend:
    def __init__(self, replies):
        self.replies, self.calls = deepcopy(replies), []
    def complete_json(self, system, payload):
        self.calls.append(payload)
        return self.replies.pop(0)


class AnswerValidationTests(unittest.TestCase):
    def test_source_gate_and_no_uncited_text_in_metadata(self):
        with self.assertRaises(GradingError):
            approved_sources(grade(False))
        source = approved_sources(grade())[0]
        self.assertNotIn("text", source["metadata"])
        self.assertEqual(source["text"], SOURCE["text"])

    def test_wrong_namespace_or_uncited_claim_rejected(self):
        for refs in ([], [{"chunk_id": "semantic-other", "span_id": "S1"}],
                     [{"chunk_id": "c1", "span_id": "invented"}]):
            draft = deepcopy(DRAFT)
            draft["claims"][0]["citations"] = refs
            with self.assertRaises(GradingError):
                validate_draft(draft, [SOURCE])

    def test_duplicate_claim_and_embedded_citation_rejected(self):
        with self.assertRaises(GradingError):
            validate_draft({"claims": DRAFT["claims"] * 2}, [SOURCE])
        draft = deepcopy(DRAFT)
        draft["claims"][0]["text"] += " [made-up-source]"
        with self.assertRaises(GradingError):
            validate_draft(draft, [SOURCE])

    def test_unsupported_claim_or_missing_coverage_blocks(self):
        for field in ("claim_checks", "coverage"):
            check = deepcopy(CHECK)
            check[field][0]["supported" if field == "claim_checks" else "covered"] = False
            self.assertFalse(validate_answer_check(check, DRAFT, PLAN)["passed"])

    def test_omitted_claim_or_requirement_check_fails_closed(self):
        for field in ("claim_checks", "coverage"):
            check = deepcopy(CHECK)
            check[field] = []
            with self.assertRaises(GradingError):
                validate_answer_check(check, DRAFT, PLAN)

    def test_coverage_cannot_rely_on_unsupported_claim(self):
        check = deepcopy(CHECK)
        check["claim_checks"][0]["supported"] = False
        check["passed"] = True
        self.assertFalse(validate_answer_check(check, DRAFT, PLAN)["passed"])

    def test_renderer_uses_source_citations(self):
        answer = render_answer(DRAFT, [SOURCE])
        self.assertIn(SOURCE["citation"], answer["text"])
        self.assertEqual(answer["sources"], [SOURCE])

    def test_verifier_only_sees_cited_spans_for_each_claim(self):
        backend = Backend([CHECK])
        engine = AnswerEngine(backend)
        engine.verify("PTO?", PLAN, DRAFT, [SOURCE, {**SOURCE, "chunk_id": "c2", "text": "Other rule"}])
        self.assertEqual(backend.calls[0]["claims"][0]["cited_sources"], [SOURCE])


class AnswerGraphTests(unittest.TestCase):
    def run_graph(self, replies, grades):
        backend = Backend(replies)
        with patch('gridaxon_hr_qa.answer_workflow.run_graded', return_value={"plan": PLAN, "grades": grades}):
            result = run_answered(object(), object(), AnswerEngine(backend), "PTO?")
        return result, backend

    def test_closed_gate_never_calls_generator(self):
        result, backend = self.run_graph([], {"fixed": grade(False)})
        self.assertFalse(backend.calls)
        self.assertEqual(result["answers"]["fixed"]["status"], "abstain")

    def test_success_after_one_revision_rechecked(self):
        failed = deepcopy(CHECK)
        failed["claim_checks"][0]["supported"] = False
        result, backend = self.run_graph([DRAFT, failed, DRAFT, CHECK], {"fixed": grade()})
        self.assertTrue(result["answers"]["fixed"]["validated"])
        self.assertEqual(len(result["attempts"]["fixed"]), 2)
        self.assertEqual([call["task"] for call in backend.calls],
                         ['generate_answer', 'check_answer', 'generate_answer', 'check_answer'])

    def test_repeated_invalid_citations_abstain_without_exposing_draft(self):
        invalid = deepcopy(DRAFT)
        invalid["claims"][0]["citations"][0]["span_id"] = "invented"
        invalid["claims"][0]["text"] = "False private balance 999 hours."
        result, backend = self.run_graph([invalid, invalid], {"fixed": grade()})
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(result["answers"]["fixed"]["status"], "abstain")
        self.assertNotIn('999', result["answers"]["fixed"]["text"])

    def test_namespaces_have_independent_generation_gates(self):
        result, backend = self.run_graph([DRAFT, CHECK], {"fixed": grade(), "semantic": grade(False)})
        self.assertTrue(result["answers"]["fixed"]["validated"])
        self.assertFalse(result["answers"]["semantic"]["validated"])
        self.assertEqual(len(backend.calls), 2)
