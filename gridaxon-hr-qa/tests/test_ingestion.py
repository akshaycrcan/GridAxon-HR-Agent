import json
from pathlib import Path
import tempfile
import unittest

from gridaxon_hr_qa.ingestion import load_policies, write_documents


POLICY = '''---
document_id: HR-001
title: Example policy
version: "1.0"
effective_date: 2026-01-01
status: current
---
# Example policy

## 1. Eligibility
Keep this exception with the rule.
'''


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "corpus").mkdir()
        (self.root / "corpus/policy.md").write_text(POLICY, encoding="utf-8")

    def test_excludes_non_corpus_and_preserves_metadata_and_source(self):
        for folder in ("archive", "evals", "references"):
            (self.root / folder).mkdir()
            (self.root / folder / "excluded.md").write_text("invalid", encoding="utf-8")
        documents = load_policies(self.root)
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["metadata"]["effective_date"], "2026-01-01")
        self.assertEqual(documents[0]["raw_markdown"],
                         (self.root / "corpus/policy.md").read_bytes().decode("utf-8"))
        self.assertNotIn("document_id:", documents[0]["text"])
        output = self.root / "output/documents.jsonl"
        write_documents(documents, output)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), documents[0])

    def test_missing_metadata_is_rejected(self):
        (self.root / "corpus/policy.md").write_text("# No metadata", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing metadata"):
            load_policies(self.root)

    def test_duplicate_current_policy_is_rejected(self):
        (self.root / "corpus/duplicate.md").write_text(POLICY, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate current"):
            load_policies(self.root)

    def test_superseded_policy_is_rejected(self):
        (self.root / "corpus/policy.md").write_text(
            POLICY.replace("status: current", "status: superseded"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Only current"):
            load_policies(self.root)


if __name__ == "__main__":
    unittest.main()
