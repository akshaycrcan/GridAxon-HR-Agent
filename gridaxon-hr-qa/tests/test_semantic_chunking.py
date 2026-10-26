from copy import deepcopy
import unittest

import numpy as np

from gridaxon_hr_qa.semantic_chunking import semantic_chunks, sentence_spans


def document(text):
    return {"metadata": {"document_id": "HR-008", "title": "Policy", "version": "2.0",
                         "effective_date": "2026-07-01", "status": "current"},
            "source_path": "corpus/policy.md", "source_sha256": "test", "text": text}


class TopicEncoder:
    identity = {"model": "test-only-topic-encoder", "revision": "1"}

    def fits(self, text):
        return len(text) <= 500

    def encode(self, texts):
        # Deterministic vectors let tests isolate grouping from model quality.
        return np.array([[0., 1.] if "Password" in text else [1., 0.] for text in texts])


class SemanticChunkingTests(unittest.TestCase):
    def test_real_semantic_decision_and_threshold_direction(self):
        source = document("PTO accrues monthly. PTO requests need notice. Password changes protect accounts.")
        low = semantic_chunks([source], TopicEncoder(), min_tokens=1, distance_threshold=0.5)
        high = semantic_chunks([source], TopicEncoder(), min_tokens=1, distance_threshold=1.5)
        self.assertEqual(len(low), 2)
        self.assertEqual(len(high), 1)
        self.assertEqual(low[0]["boundary_reason"], "semantic_break")
        self.assertIn("Password", low[1]["text"])

    def test_coverage_metadata_unicode_limits_and_reproducibility(self):
        body = "# Policy\nIntroduction.\n## 1. PTO\n" + "PTO café 👩🏽‍💻 exception. " * 30
        body += "\n## 2. Security\n" + "Password controls are required. " * 20
        source = document(body)
        before = deepcopy(source)
        chunks = semantic_chunks([source], TopicEncoder(), min_tokens=10, max_tokens=30)
        self.assertEqual("".join(c["text"] for c in chunks), body)
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))
        self.assertEqual(chunks, semantic_chunks([source], TopicEncoder(), 10, 30))
        for chunk in chunks:
            self.assertEqual(chunk["metadata"], source["metadata"])
            self.assertEqual(chunk["text"], body[chunk["start_char"]:chunk["end_char"]])
            self.assertLessEqual(chunk["token_count"], 30)
            self.assertGreaterEqual(chunk["start_char"], chunk["section_start_char"])
            self.assertLessEqual(chunk["end_char"], chunk["section_end_char"])
        self.assertEqual(source, before)

    def test_long_sentence_is_split_before_embedding(self):
        class LimitedEncoder(TopicEncoder):
            def fits(self, text):
                return len(text) <= 50

            def encode(self, texts):
                assert all(self.fits(text) for text in texts)
                return super().encode(texts)

        body = "Eligibility " * 100 + "."
        chunks = semantic_chunks([document(body)], LimitedEncoder(), 10, 25)
        self.assertEqual("".join(c["text"] for c in chunks), body)
        self.assertTrue(all(c["token_count"] <= 25 for c in chunks))

    def test_abbreviation_and_decimal_sentence_boundaries(self):
        body = "Dr. Smith earns 4.62 hours. PTO is available."
        spans = list(sentence_spans(body))
        self.assertEqual(len(spans), 2)
        self.assertIn("4.62", body[spans[0][0]:spans[0][1]])

    def test_short_section_and_settings(self):
        chunks = semantic_chunks([document("## 1. Rule\nShort.")], TopicEncoder())
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["boundary_reason"], "section_end")
        for minimum, maximum, threshold in [(0, 300, .5), (400, 300, .5),
                                            (80, 300, float("nan")), (80, 300, 3)]:
            with self.assertRaises(ValueError):
                semantic_chunks([], TopicEncoder(), minimum, maximum, threshold)

    def test_bad_embeddings_rejected(self):
        class BrokenEncoder(TopicEncoder):
            def encode(self, texts):
                return np.zeros((len(texts), 2))

        with self.assertRaisesRegex(ValueError, "zero-length"):
            semantic_chunks([document("A policy.")], BrokenEncoder())

    def test_semantic_break_does_not_leave_tiny_tail(self):
        text = "PTO accrues for eligible employees each pay period. " * 4 + "Password rules apply."
        chunks = semantic_chunks([document(text)], TopicEncoder(), min_tokens=20, max_tokens=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], text)
