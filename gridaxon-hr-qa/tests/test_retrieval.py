from copy import deepcopy
import unittest

import numpy as np

from gridaxon_hr_qa.retrieval import Retriever
from gridaxon_hr_qa.routing import build_retrieval_graph, choose_route
from gridaxon_hr_qa.retrieval_checks import score_evidence


class Encoder:
    identity = {"model": "test", "revision": "commit", "dimension": 2}

    def encode_query(self, question):
        return np.array([1., 0.])

    def encode(self, texts):
        return np.array([[1., 0.] for _ in texts])


def metadata(document_id="HR-008", method="fixed"):
    return {"document_id": document_id, "title": "PTO", "version": "2.0", "section": "1",
            "section_title": "Eligibility", "effective_date": "2026-07-01", "status": "current",
            "text": "Eligible employees accrue PTO.", "source_path": "corpus/policy.md",
            "start_char": 0, "end_char": 29, "build_id": "build",
            "chunking_method": method, "embedding_revision": "commit", "embedding_model": "test"}


class Client:
    def __init__(self):
        self.calls = []
        self.corrupt = False

    def request(self, method, host, path, payload):
        self.calls.append(payload)
        kind = payload["namespace"].split('-')[0]
        meta = metadata(method=kind)
        if self.corrupt:
            meta["build_id"] = "wrong"
        match = {"id": kind + '-chunk', "score": .8, "metadata": meta}
        return {"namespace": payload["namespace"], "matches": [match, match]}


def make_retriever():
    active = {"embedding": Encoder.identity, "host": "test.svc.pinecone.io", "build_id": "build",
              "datasets": {"fixed": {"namespace": "fixed-build"}, "semantic": {"namespace": "semantic-build"}}}
    datasets = {"fixed": [{"metadata": metadata()}]}
    return Retriever(active, datasets, Encoder(), Client())


class RetrievalTests(unittest.TestCase):
    def test_both_namespaces_filters_citations_and_deduplication(self):
        retriever = make_retriever()
        result = retriever.retrieve("How much PTO?", as_of="2026-10-24")
        self.assertEqual(set(result), {"fixed", "semantic"})
        self.assertEqual(len(result["fixed"]), 1)
        self.assertEqual(result["fixed"][0]["citation"], "[HR-008 v2.0 section 1]")
        self.assertEqual([call["namespace"] for call in retriever.client.calls], ["fixed-build", "semantic-build"])
        for call in retriever.client.calls:
            self.assertTrue(call["includeMetadata"])
            self.assertFalse(call["includeValues"])
            self.assertIn({"build_id": {"$eq": "build"}}, call["filter"]["$and"])

    def test_effective_date_excludes_future_policy_without_remote_request(self):
        retriever = make_retriever()
        self.assertEqual(retriever.retrieve("PTO?", as_of="2026-01-01"), {"fixed": [], "semantic": []})
        self.assertEqual(retriever.client.calls, [])

    def test_invalid_query_settings(self):
        retriever = make_retriever()
        with self.assertRaises(ValueError):
            retriever.embed_question(" ")
        for top_k in (0, 51, True):
            with self.assertRaises(ValueError):
                retriever.query([1, 0], top_k=top_k)
        with self.assertRaises(ValueError):
            retriever.query([1, 0], document_ids=["HR-999"])
        with self.assertRaises(ValueError):
            retriever.query([1, 0, 0])

    def test_rejects_incompatible_remote_metadata(self):
        retriever = make_retriever()
        retriever.client.corrupt = True
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            retriever.retrieve("PTO?")

    def test_embedding_identity_mismatch(self):
        retriever = make_retriever()
        wrong = deepcopy(retriever.active)
        wrong["embedding"]["revision"] = "other"
        with self.assertRaisesRegex(ValueError, "exactly"):
            Retriever(wrong, {"fixed": []}, Encoder(), Client())


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.ranked = [{"document_id": f"HR-{i:03d}", "score": score}
                       for i, score in enumerate([.8, .7, .65, .4], 1)]
        self.ids = [item["document_id"] for item in self.ranked]

    def test_multiple_document_route(self):
        route = choose_route("A policy question", self.ranked, self.ids, self.ids)
        self.assertEqual(route["mode"], "routed")
        self.assertEqual(route["document_ids"], self.ids[:3])

    def test_ambiguous_rankings_search_all(self):
        ranked = [{**item, "score": .5} for item in self.ranked]
        self.assertEqual(choose_route("Question", ranked, self.ids, self.ids)["mode"], "broad")

    def test_explicit_multi_policy_scope_and_unknown_reference(self):
        route = choose_route("Compare HR-001 and HR-004", self.ranked, self.ids, self.ids)
        self.assertEqual(route["document_ids"], ["HR-001", "HR-004"])
        self.assertEqual(route["mode"], "explicit")
        with self.assertRaises(ValueError):
            choose_route("HR-999?", self.ranked, self.ids, self.ids)

    def test_real_graph_falls_back_once(self):
        retriever = make_retriever()
        result = build_retrieval_graph(retriever).invoke({"question": "PTO?", "top_k": 5})
        self.assertEqual(result["trace"], ["embed_question", "route_documents", "retrieve", "broaden_search"])
        self.assertEqual(len(retriever.client.calls), 4)
        self.assertEqual(set(result["results"]), {"fixed", "semantic"})

    def test_explicit_scope_not_broadened(self):
        retriever = make_retriever()
        result = build_retrieval_graph(retriever).invoke({"question": "HR-008 PTO?", "top_k": 5})
        self.assertNotIn("broaden_search", result["trace"])
        self.assertEqual(len(retriever.client.calls), 2)

    def test_no_eligible_policies_returns_empty(self):
        retriever = make_retriever()
        result = build_retrieval_graph(retriever).invoke({"question": "PTO?", "as_of": "2025-01-01"})
        self.assertEqual(result["results"], {"fixed": [], "semantic": []})
        self.assertEqual(retriever.client.calls, [])


class EvidenceLabelTests(unittest.TestCase):
    def test_verified_alternative_retains_strict_reference_miss(self):
        result = score_evidence([("HR-006", "1.0", "2")],
                                [{"metadata": {"document_id": "HR-006", "version": "1.0", "section": "4"}}])
        self.assertTrue(result["all_sections_found"])
        self.assertFalse(result["all_exact_sections_found"])
        self.assertEqual(len(result["accepted_alternatives"]), 1)

    def test_alternative_does_not_excuse_missing_second_policy(self):
        result = score_evidence([("HR-006", "1.0", "2"), ("HR-005", "1.0", "1")],
                                [{"metadata": {"document_id": "HR-006", "version": "1.0", "section": "4"}}])
        self.assertFalse(result["all_sections_found"])
        self.assertEqual(result["missing_sections"], [["HR-005", "1.0", "1"]])
