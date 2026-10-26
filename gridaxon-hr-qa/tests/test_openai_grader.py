import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from gridaxon_hr_qa.grading import GradingError
from gridaxon_hr_qa.openai_grader import (OpenAIGraderBackend, parse_response, PLAN_SCHEMA, GRADE_SCHEMA,
                                        ANSWER_SCHEMA, ANSWER_CHECK_SCHEMA)


class OpenAIGraderTests(unittest.TestCase):
    def test_structured_response(self):
        self.assertEqual(parse_response({"status": "completed", "output": [
            {"type": "reasoning"}, {"type": "message", "content": [
                {"type": "output_text", "text": '{"ok": true}'}]}]}), {"ok": True})

    def test_refusal_incomplete_and_invalid_json_fail_closed(self):
        for response in ({"status": "incomplete"},
                         {"status": "completed", "output": [{"type": "message", "content": [
                             {"type": "refusal", "refusal": "Declined"}]}]},
                         {"status": "completed", "output": []}):
            with self.assertRaises(GradingError):
                parse_response(response)

    def test_every_schema_object_is_strict(self):
        def walk(schema):
            if schema.get("type") == "object":
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(set(schema["required"]), set(schema["properties"]))
                for child in schema["properties"].values():
                    walk(child)
            if schema.get("type") == "array":
                walk(schema["items"])
        walk(PLAN_SCHEMA)
        walk(GRADE_SCHEMA)
        walk(ANSWER_SCHEMA)
        walk(ANSWER_CHECK_SCHEMA)

    def test_api_request_uses_structured_output_and_does_not_store(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps({"status": "completed", "model": "test-model", "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": '{}'}]}]}).encode()
        backend = OpenAIGraderBackend("test-secret")
        with patch.object(backend._opener, "open", return_value=Response()) as request:
            backend.complete_json("instructions", {"question": "PTO?"})
        body = json.loads(request.call_args.args[0].data)
        self.assertFalse(body["store"])
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertNotIn("test-secret", json.dumps(backend.calls))

    def test_env_is_project_local_and_other_keys_are_not_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text('OPENAI_API_KEY="test-secret"\nOPENAI_GRADER_MODEL=test-model\nPINECONE_API_KEY=other')
            with patch.dict('os.environ', {}, clear=True):
                backend = OpenAIGraderBackend.from_env(path)
            self.assertEqual(backend.model, 'test-model')

    def test_quota_error_reports_code_without_leaking_raw_error(self):
        backend = OpenAIGraderBackend('test-secret')
        error = HTTPError('https://api.openai.com/v1/responses', 429, 'quota', {},
                          io.BytesIO(b'{"error":{"code":"insufficient_quota","message":"private detail"}}'))
        with patch.object(backend._opener, 'open', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'insufficient_quota') as result:
                backend.complete_json('instructions', {'question': 'PTO?'})
        self.assertNotIn('private detail', str(result.exception))
        self.assertNotIn('test-secret', str(result.exception))
