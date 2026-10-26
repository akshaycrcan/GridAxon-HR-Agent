import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs, urlsplit

import numpy as np

from gridaxon_hr_qa.embeddings import prepare_embeddings, load_build, validate_vectors
from gridaxon_hr_qa.ingestion import write_documents
from gridaxon_hr_qa.pinecone_store import ensure_index, upload_build, verify_records


class Encoder:
    identity = {"model": "test", "revision": "test", "dimension": 3}

    def encode(self, texts):
        return np.array([[1, 2, 3] for _ in texts])


class FakePinecone:
    def __init__(self):
        self.records = {}

    def request(self, method, host, path, payload=None, allow_missing=False):
        if path.startswith('/indexes/'):
            return {"dimension": 3, "metric": "cosine", "status": {"ready": True},
                    "host": "example.svc.pinecone.io"}
        if path == '/vectors/upsert':
            self.records.setdefault(payload['namespace'], {}).update({r['id']: r for r in payload['vectors']})
            return {"upsertedCount": len(payload['vectors'])}
        if path.startswith('/vectors/fetch?'):
            query = parse_qs(urlsplit(path).query)
            records = self.records.get(query['namespace'][0], {})
            return {'vectors': {i: records[i] for i in query['ids'] if i in records}}
        raise AssertionError(path)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"PINECONE_API_KEY": "test-key", "PINECONE_INDEX": "test"}
        for method in ('fixed', 'semantic'):
            chunk = {"chunk_id": method + '-1', "metadata": {"document_id": "HR-008", "title": "PTO", "version": "2.0"},
                     "section": "1", "section_title": "PTO", "source_path": "corpus/policy.md",
                     "source_sha256": "same", "start_char": 0, "end_char": 4,
                     "section_start_char": 0, "section_end_char": 4,
                     "chunk_index_in_section": 0, "token_count": 1, "text": "Rule",
                     "chunking": {"method": method}}
            write_documents([chunk], self.root / f'chunks_{method}.jsonl')

    def test_prepare_upload_verify_and_idempotence(self):
        manifest = prepare_embeddings(self.root, Encoder())
        _, datasets = load_build(self.root)
        self.assertTrue(np.isclose(np.linalg.norm(datasets['fixed'][0]['values']), 1))
        self.assertNotEqual(manifest['datasets']['fixed']['namespace'], manifest['datasets']['semantic']['namespace'])
        client = FakePinecone()
        upload_build(self.root, self.config, client=client)
        upload_build(self.root, self.config, client=client)
        self.assertEqual(sum(len(v) for v in client.records.values()), 2)
        active = (self.root / 'pinecone_active.json').read_text()
        self.assertNotIn('test-key', active)
        self.assertTrue(json.loads(active)['verified'])

    def test_checksum_failure_prevents_upload(self):
        manifest = prepare_embeddings(self.root, Encoder())
        path = self.root / manifest['datasets']['fixed']['file']
        path.write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'Checksum'):
            upload_build(self.root, self.config, client=FakePinecone())
        self.assertFalse((self.root / 'pinecone_active.json').exists())

    def test_invalid_vectors(self):
        for values in ([[0, 0, 0]], [[1, float('nan'), 3]], [[1, 2]]):
            with self.assertRaises(ValueError):
                validate_vectors(values, 1, 3)

    def test_wrong_index_dimension(self):
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            ensure_index(FakePinecone(), self.config, 384)

    def test_missing_remote_record_fails_verification(self):
        with self.assertRaisesRegex(RuntimeError, 'Verification incomplete'):
            verify_records(FakePinecone(), 'example.svc.pinecone.io', 'fixed',
                           [{'id': 'missing', 'values': [1], 'metadata': {}}], timeout=0)

    def test_different_corpora_rejected(self):
        path = self.root / 'chunks_semantic.jsonl'
        chunk = json.loads(path.read_text())
        chunk['source_sha256'] = 'different'
        write_documents([chunk], path)
        with self.assertRaisesRegex(ValueError, 'same source versions'):
            prepare_embeddings(self.root, Encoder())
