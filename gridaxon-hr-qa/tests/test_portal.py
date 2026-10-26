import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from gridaxon_hr_qa.portal import public_answer, highlighted_document, linked_answer, ROOT


DOC = {"metadata": {"document_id": "HR-008", "title": "PTO policy", "version": "2.0", "effective_date": "2026-01-01"},
       "text": "# PTO\n120 hours annually. <script>bad()</script>", "source_sha256": "abc", "raw_markdown": "# PTO\n120 hours annually."}
SOURCE = {"citation": "[HR-008 v2.0 section 1]", "text": "120 hours annually.",
          "metadata": {**DOC["metadata"], "source_sha256": "abc", "start_char": 6, "section": "1"},
          "quote_start_char": 0, "quote_end_char": 19}
ANSWER = {"status": "answered", "text": "120 hours annually. [HR-008 v2.0 section 1]", "sources": [SOURCE]}


class PortalLogicTests(unittest.TestCase):
    def test_rejected_draft_cannot_reach_ui(self):
        result = {"answers": {"fixed": {"status": "abstain", "validated": False, "text": "999 invented hours", "sources": [SOURCE]}},
                  "drafts": {"fixed": "999 invented hours"}}
        public = public_answer(result)
        self.assertNotIn("999", public["text"])
        self.assertEqual(public["sources"], [])

    def test_answer_requires_check(self):
        with self.assertRaises(ValueError):
            public_answer({"answers": {"fixed": {**ANSWER, "validated": True}}})

    def test_exact_highlight_and_html_escape(self):
        html = highlighted_document(DOC, [SOURCE])
        self.assertIn("<mark>120 hours annually.</mark>", html)
        self.assertNotIn("<script>", html)
        stale = {**SOURCE, "metadata": {**SOURCE["metadata"], "source_sha256": "old"}}
        self.assertNotIn("<mark>", highlighted_document(DOC, [stale]))

    def test_citation_links_are_app_owned(self):
        self.assertIn("[[1]](#source-1)", linked_answer(ANSWER))
        self.assertIn(r"\[evil\]", linked_answer({**ANSWER, "text": "[evil](https://example.com)"}))


class PortalAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        (folder / "documents.jsonl").write_text(json.dumps(DOC) + "\n", encoding="utf-8")
        self.storage = patch("gridaxon_hr_qa.portal.STORAGE", folder)
        self.storage.start()
        self.addCleanup(self.storage.stop)
        self.app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=15)

    def test_browse_search_and_suggestion_without_api(self):
        with patch("gridaxon_hr_qa.portal.ask_policy") as ask:
            at = self.app.run()
            self.assertFalse(at.exception)
            at.text_input(key="policy_search").set_value("no such policy").run()
            self.assertTrue(any("No policies match" in item.value for item in at.caption))
            at.button(key="suggest_PTO & leave").click().run()
            self.assertIn("PTO", at.text_input(key="question").value)
            ask.assert_not_called()

    def test_answer_highlight_browse_preserves_answer_no_repeat_charge(self):
        with patch("gridaxon_hr_qa.portal.ask_policy", return_value=ANSWER) as ask:
            at = self.app.run()
            at.text_input(key="question").set_value("How much PTO?")
            at.form_submit_button[0].click().run()
            self.assertFalse(at.exception)
            self.assertIn("#source-1", " ".join(m.value for m in at.markdown))
            self.assertEqual(at.button(key="doc_HR-008").proto.type, "primary")
            at.text_input(key="policy_search").set_value("PTO").run()
            at.button(key="source_1").click().run()
            self.assertFalse(at.exception)
            ask.assert_called_once()

    def test_failure_clears_previous_answer_and_redacts_exception(self):
        with patch("gridaxon_hr_qa.portal.ask_policy", side_effect=[ANSWER, RuntimeError("secret-token")]):
            at = self.app.run()
            at.text_input(key="question").set_value("PTO?")
            at.form_submit_button[0].click().run()
            at.text_input(key="question").set_value("Other question?")
            at.form_submit_button[0].click().run()
            self.assertFalse(at.exception)
            self.assertIsNone(at.session_state.answer)
            self.assertNotIn("secret-token", " ".join(e.value for e in at.error))
            self.assertEqual(at.button(key="doc_HR-008").proto.type, "tertiary")

    def test_empty_question_no_api(self):
        with patch("gridaxon_hr_qa.portal.ask_policy") as ask:
            at = self.app.run()
            at.form_submit_button[0].click().run()
            self.assertTrue(at.error)
            ask.assert_not_called()
