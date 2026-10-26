"""OpenAI Responses structured-output adapter; credentials stay out of reports."""

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from gridaxon_hr_qa.grading import GradingError
from gridaxon_hr_qa.pinecone_store import NoRedirect, read_env_values


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array(items):
    return {"type": "array", "items": items}


TEXT = {"type": "string"}
PLAN_SCHEMA = obj({"requirements": array(obj({"id": TEXT, "description": TEXT})),
                   "clarification_question": TEXT})
GRADE_SCHEMA = obj({
    "relevance": array(obj({"chunk_id": TEXT, "relevant": {"type": "boolean"}, "reason": TEXT})),
    "coverage": array(obj({"requirement_id": TEXT,
                           "status": {"type": "string", "enum": ["supported", "partial", "missing"]},
                           "reason": TEXT, "evidence": array(obj({"chunk_id": TEXT, "span_id": TEXT}))})),
    "conflicts": array(obj({"description": TEXT, "chunk_ids": array(TEXT)})),
    "clarification_question": TEXT,
})
ANSWER_SCHEMA = obj({"claims": array(obj({
    "claim_id": TEXT, "text": TEXT,
    "citations": array(obj({"chunk_id": TEXT, "span_id": TEXT})),
}))})
ANSWER_CHECK_SCHEMA = obj({
    "claim_checks": array(obj({"claim_id": TEXT, "supported": {"type": "boolean"}, "reason": TEXT})),
    "coverage": array(obj({"requirement_id": TEXT, "covered": {"type": "boolean"},
                           "claim_ids": array(TEXT), "reason": TEXT})),
})


def parse_response(response):
    if response.get("status") != "completed":
        raise GradingError("OpenAI grading response was incomplete; generation remains blocked")
    texts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "refusal":
                raise GradingError("OpenAI declined this grading request; generation remains blocked")
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    try:
        result = json.loads("".join(texts))
    except (ValueError, TypeError):
        raise GradingError("OpenAI returned invalid grading JSON") from None
    if not isinstance(result, dict):
        raise GradingError("OpenAI grading response must be an object")
    return result


class OpenAIGraderBackend:
    def __init__(self, api_key, model="gpt-5-mini"):
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in this project's .env")
        self._key, self.model = api_key, model
        self.calls = []
        self._opener = build_opener(NoRedirect)

    @classmethod
    def from_env(cls, env_file=Path(".env"), model=None):
        config = read_env_values(env_file, {"OPENAI_API_KEY", "OPENAI_GRADER_MODEL"})
        return cls(config.get("OPENAI_API_KEY"), model or config.get("OPENAI_GRADER_MODEL") or "gpt-5-mini")

    def complete_json(self, system, payload):
        is_grade = "passages" in payload
        stage = "grade" if is_grade else "plan"
        schema = GRADE_SCHEMA if is_grade else PLAN_SCHEMA
        if payload.get("task") == "generate_answer":
            stage, schema = "generate_answer", ANSWER_SCHEMA
        elif payload.get("task") == "check_answer":
            stage, schema = "check_answer", ANSWER_CHECK_SCHEMA
        body = {"model": self.model, "store": False, "max_output_tokens": 6000,
                "instructions": system + "\nKeep assessment reasons concise.",
                "input": json.dumps(payload, ensure_ascii=False),
                "text": {"format": {"type": "json_schema", "name": stage,
                                    "strict": True, "schema": schema}}}
        if self.model.startswith("gpt-5"):
            body["reasoning"] = {"effort": "low"}
        request = Request("https://api.openai.com/v1/responses", method="POST",
                          data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                                                  "Authorization": f"Bearer {self._key}"})
        try:
            with self._opener.open(request, timeout=60) as response:
                result = json.load(response)
        except HTTPError as exc:
            hints = {400: "request/model configuration rejected", 401: "check OPENAI_API_KEY",
                     403: "check model/project permissions", 404: "check OPENAI_GRADER_MODEL",
                     429: "check API billing, quota, or rate limits"}
            if exc.code == 429:
                try:
                    code = json.loads(exc.read()).get("error", {}).get("code")
                except (ValueError, AttributeError):
                    code = None
                if code == "insufficient_quota":
                    hints[429] = "insufficient_quota: add API credits or resolve the project's billing/quota limit"
                elif code == "rate_limit_exceeded":
                    hints[429] = "rate_limit_exceeded: wait before retrying or reduce request rate"
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {hints.get(exc.code, 'API request failed')}; "
                               "generation remains blocked") from None
        except (URLError, TimeoutError):
            raise RuntimeError("OpenAI connection failed or timed out; generation remains blocked") from None
        self.calls.append({"stage": stage, "model": result.get("model", self.model),
                           "response_id": result.get("id"), "usage": result.get("usage", {})})
        return parse_response(result)
