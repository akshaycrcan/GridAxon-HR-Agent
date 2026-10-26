"""Pinecone vector REST API; no SDK dependency or credentials in artifacts."""

import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler

import numpy as np

from gridaxon_hr_qa.embeddings import load_build, write_json


def read_env_values(env_file, allowed):
    """Read simple KEY=value lines; do not execute or interpolate file contents."""
    values = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, separator, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if separator and key in allowed:
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                else:
                    value = value.split(" #", 1)[0].rstrip()
                values[key] = value
    values.update({key: os.environ[key] for key in allowed if key in os.environ})
    return values


def settings(env_file=Path(".env")):
    values = read_env_values(env_file, {"PINECONE_API_KEY", "PINECONE_INDEX", "PINECONE_CLOUD", "PINECONE_REGION"})
    if not values.get("PINECONE_API_KEY"):
        raise ValueError("Set PINECONE_API_KEY in .env or the environment")
    values.setdefault("PINECONE_INDEX", "gridaxon-hr-qa")
    values.setdefault("PINECONE_CLOUD", "aws")
    values.setdefault("PINECONE_REGION", "us-east-1")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,43}[a-z0-9]|[a-z0-9]", values["PINECONE_INDEX"]):
        raise ValueError("Invalid Pinecone index name")
    return values


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PineconeREST:
    def __init__(self, api_key):
        self._key = api_key
        self._opener = build_opener(NoRedirect)

    def request(self, method, host, path, payload=None, allow_missing=False):
        if not re.fullmatch(r"[a-zA-Z0-9.-]+\.pinecone\.io", host):
            raise ValueError("Refusing credentials for a non-Pinecone host")
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(f"https://{host}{path}", data=body, method=method,
                          headers={"Api-Key": self._key, "Content-Type": "application/json",
                                   "X-Pinecone-Api-Version": "2025-10"})
        for attempt in range(3):
            try:
                with self._opener.open(request, timeout=30) as response:
                    return json.load(response)
            except HTTPError as exc:
                if allow_missing and exc.code == 404:
                    return None
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                # Never print API headers, the secret, or raw server error bodies.
                hint = {401: "check your API key", 403: "check project permissions",
                        409: "index already exists; rerun", 400: "check index settings and account region limits"}
                raise RuntimeError(f"Pinecone HTTP {exc.code}: {hint.get(exc.code, 'request failed')}") from None
            except (URLError, TimeoutError):
                raise RuntimeError("Pinecone connection failed; check network access and retry") from None


def ensure_index(client, config, dimension, create=False, timeout=90):
    name = config["PINECONE_INDEX"]
    path = f"/indexes/{name}"
    description = client.request("GET", "api.pinecone.io", path, allow_missing=True)
    if description is None:
        if not create:
            raise ValueError("Index does not exist; use --create-index to create it")
        description = client.request("POST", "api.pinecone.io", "/indexes", {
            "name": name, "vector_type": "dense", "dimension": dimension, "metric": "cosine",
            "spec": {"serverless": {"cloud": config["PINECONE_CLOUD"],
                                    "region": config["PINECONE_REGION"]}},
            "deletion_protection": "enabled",
        })
    deadline = time.monotonic() + timeout
    while True:
        if description.get("dimension") != dimension or description.get("metric") != "cosine":
            raise ValueError(f"Index must use {dimension} dimensions and cosine similarity")
        if description.get("status", {}).get("ready"):
            return description["host"]
        if time.monotonic() >= deadline:
            raise RuntimeError("Index not ready before timeout; rerun indexing shortly")
        time.sleep(2)
        description = client.request("GET", "api.pinecone.io", path)


def verify_records(client, host, namespace, records, timeout=60):
    pending = {record["id"]: record for record in records}
    deadline = time.monotonic() + timeout
    while pending:
        ids = list(pending)
        for start in range(0, len(ids), 20):
            query = urlencode({"ids": ids[start:start + 20], "namespace": namespace}, doseq=True)
            fetched = client.request("GET", host, f"/vectors/fetch?{query}").get("vectors", {})
            for record_id, remote in fetched.items():
                if record_id not in pending:
                    continue
                expected = pending[record_id]
                values = remote.get("values", [])
                if (remote.get("metadata") == expected["metadata"]
                        and len(values) == len(expected["values"])
                        and np.allclose(values, expected["values"], atol=1e-6, rtol=1e-5)):
                    del pending[record_id]
        if pending:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Verification incomplete for {len(pending)} vectors in {namespace}; "
                                   "rerun safely after Pinecone finishes indexing")
            time.sleep(2)


def upload_build(storage, config, create=False, client=None):
    manifest, datasets = load_build(storage)
    client = client or PineconeREST(config["PINECONE_API_KEY"])
    host = ensure_index(client, config, manifest["embedding"]["dimension"], create)
    for method, records in datasets.items():
        namespace = manifest["datasets"][method]["namespace"]
        print(f"Uploading {len(records)} {method} vectors to {namespace}...", flush=True)
        for start in range(0, len(records), 50):
            batch = records[start:start + 50]
            result = client.request("POST", host, "/vectors/upsert", {"namespace": namespace, "vectors": batch})
            if result.get("upsertedCount") != len(batch):
                raise RuntimeError("Pinecone did not acknowledge the full batch")
        verify_records(client, host, namespace, records)
        print(f"Verified all {len(records)} {method} vectors and their metadata.", flush=True)
    active = {**manifest, "index": config["PINECONE_INDEX"], "host": host, "verified": True}
    write_json(storage / "pinecone_active.json", active)
    return active
