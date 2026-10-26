from copy import deepcopy
import unittest

from gridaxon_hr_qa.chunking import fixed_chunks, sections


def document(text):
    return {"metadata": {"document_id": "HR-008", "version": "2.0",
                         "effective_date": "2026-07-01", "status": "current"},
            "text": text, "source_path": "corpus/policy.md", "source_sha256": "abc"}


class ChunkingTests(unittest.TestCase):
    def test_full_coverage_exact_offsets_and_section_isolation(self):
        body = "# Policy\nIntroduction.\n\n## 1. PTO\n" + "Policy exception. " * 50
        body += "\n## 2. Leave\n" + "Leave rules. " * 40
        source = document(body)
        original = deepcopy(source)
        chunks = fixed_chunks([source], 30, 8)
        covered = set()
        for chunk in chunks:
            start, end = chunk["start_char"], chunk["end_char"]
            self.assertEqual(chunk["text"], body[start:end])
            self.assertEqual(chunk["metadata"], source["metadata"])
            self.assertLessEqual(chunk["token_count"], 30)
            self.assertGreaterEqual(start, chunk["section_start_char"])
            self.assertLessEqual(end, chunk["section_end_char"])
            covered.update(range(start, end))
        self.assertEqual(covered, set(range(len(body))))
        self.assertEqual({c["section"] for c in chunks}, {"intro", "1", "2"})
        self.assertEqual(source, original)
        same_section = [c for c in chunks if c["section"] == "1"]
        self.assertLess(same_section[1]["start_char"], same_section[0]["end_char"])

    def test_unicode_and_special_token_literal_preserved(self):
        body = "## 1. 日本語\n" + "Café 👩🏽‍💻 — PTO <|endoftext|> " * 20
        chunks = fixed_chunks([document(body)], 20, 4)
        covered = set()
        for chunk in chunks:
            self.assertNotIn("\ufffd", chunk["text"])
            self.assertEqual(chunk["text"], body[chunk["start_char"]:chunk["end_char"]])
            covered.update(range(chunk["start_char"], chunk["end_char"]))
        self.assertEqual(covered, set(range(len(body))))

    def test_ids_reproducible_and_configuration_sensitive(self):
        source = document("## 1. Rule\nA short policy.")
        first = fixed_chunks([source], 30, 5)
        self.assertEqual(first, fixed_chunks([source], 30, 5))
        self.assertNotEqual(first[0]["chunk_id"], fixed_chunks([source], 40, 5)[0]["chunk_id"])

    def test_invalid_settings(self):
        for size, overlap in [(0, 0), (10, -1), (10, 10), (10, 11)]:
            with self.assertRaises(ValueError):
                fixed_chunks([], size, overlap)

    def test_fenced_headings_are_not_sections(self):
        body = "## 1. Actual\n```md\n## 2. Example only\n```\n## 3. Actual\nText"
        self.assertEqual([s[0] for s in sections(body)], ["1", "3"])

    def test_no_heading_and_zero_overlap(self):
        body = "Plain policy. " * 30
        chunks = fixed_chunks([document(body)], 20, 0)
        self.assertEqual("".join(c["text"] for c in chunks), body)
        self.assertEqual({c["section"] for c in chunks}, {"intro"})
