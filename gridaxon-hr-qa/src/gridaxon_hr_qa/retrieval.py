"""Read-only Pinecone top-k retrieval using the verified active build."""

from datetime import date
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from gridaxon_hr_qa.embeddings import ChunkEncoder, read_jsonl
from gridaxon_hr_qa.pinecone_store import PineconeREST, settings


def load_active(storage):
    active = json.loads((storage / "pinecone_active.json").read_text(encoding="utf-8"))
    if active.get("verified") is not True or active.get("metric") != "cosine":
        raise ValueError("Run and verify index-chunks before retrieval")
    if set(active["datasets"]) != {"fixed", "semantic"}:
        raise ValueError("Both active namespaces are required")
    datasets = {}
    for method, dataset in active["datasets"].items():
        if dataset["namespace"] != f"{method}-{active['build_id']}":
            raise ValueError("Active namespace/build mismatch")
        path = storage / dataset["file"]
        if path.resolve().parent != storage.resolve():
            raise ValueError("Active vector artifact must be inside storage")
        if sha256(path.read_bytes()).hexdigest() != dataset["sha256"]:
            raise ValueError("Active vector artifact checksum mismatch")
        records = read_jsonl(path)
        if len(records) != dataset["count"]:
            raise ValueError("Active vector count mismatch")
        datasets[method] = records
    return active, datasets


class Retriever:
    def __init__(self, active, datasets, encoder, client):
        if encoder.identity != active["embedding"]:
            raise ValueError("Query embedding configuration must match the indexed model exactly")
        self.active, self.encoder, self.client = active, encoder, client
        self.catalog = {}
        for record in datasets["fixed"]:
            meta = record["metadata"]
            entry = self.catalog.setdefault(meta["document_id"], {
                "document_id": meta["document_id"], "title": meta["title"],
                "effective_date": meta["effective_date"], "status": meta["status"],
                "version": meta["version"], "headings": set(),
            })
            if meta["section"] != "intro":
                entry["headings"].add(meta["section_title"])
        self._catalog_vectors = None

    @classmethod
    def from_storage(cls, storage=Path("storage"), env_file=Path(".env")):
        active, datasets = load_active(storage)
        config = settings(env_file)
        if config["PINECONE_INDEX"] != active["index"]:
            raise ValueError("Configured Pinecone index differs from the verified active index")
        encoder = ChunkEncoder(revision=active["embedding"]["revision"])
        return cls(active, datasets, encoder, PineconeREST(config["PINECONE_API_KEY"]))

    def eligible_documents(self, as_of=None):
        as_of = date.fromisoformat(as_of) if as_of else date.today()
        return sorted(document_id for document_id, meta in self.catalog.items()
                      if meta["status"] == "current"
                      and date.fromisoformat(meta["effective_date"]) <= as_of)

    def embed_question(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must not be empty")
        return self.encoder.encode_query(question.strip()).tolist()

    def catalog_scores(self, vector, eligible):
        if self._catalog_vectors is None:
            ids = sorted(self.catalog)
            descriptions = [self.catalog[i]["title"] + "\n" +
                            "\n".join(sorted(self.catalog[i]["headings"])) for i in ids]
            vectors = self.encoder.encode(descriptions)
            self._catalog_vectors = dict(zip(ids, vectors, strict=True))
        ranked = [{"document_id": i, "title": self.catalog[i]["title"],
                   "score": float(np.dot(self._catalog_vectors[i], vector))} for i in eligible]
        return sorted(ranked, key=lambda item: (-item["score"], item["document_id"]))

    def query(self, vector, method="fixed", top_k=5, document_ids=None, as_of=None):
        if method not in ("fixed", "semantic"):
            raise ValueError("Method must be fixed or semantic")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise ValueError("top-k must be an integer between 1 and 50")
        vector = np.asarray(vector, dtype=float)
        if (vector.shape != (self.active["embedding"]["dimension"],)
                or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0):
            raise ValueError("Invalid query vector")
        vector = vector / np.linalg.norm(vector)
        eligible = set(self.eligible_documents(as_of))
        if document_ids is not None:
            unknown = set(document_ids) - self.catalog.keys()
            if unknown:
                raise ValueError(f"Unknown document IDs: {', '.join(sorted(unknown))}")
            eligible &= set(document_ids)
        namespace = self.active["datasets"][method]["namespace"]
        if not eligible:
            return []
        payload = {"namespace": namespace, "vector": vector.tolist(), "topK": top_k,
                   "includeMetadata": True, "includeValues": False,
                   "filter": {"$and": [
                       {"build_id": {"$eq": self.active["build_id"]}},
                       {"status": {"$eq": "current"}},
                       {"document_id": {"$in": sorted(eligible)}},
                   ]}}
        response = self.client.request("POST", self.active["host"], "/query", payload)
        if response.get("namespace", namespace) != namespace:
            raise RuntimeError("Pinecone returned a different namespace")
        results, seen = [], set()
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            required = ("document_id", "version", "section", "title", "text", "source_path",
                        "start_char", "end_char", "effective_date")
            if (any(key not in meta for key in required)
                    or meta.get("build_id") != self.active["build_id"]
                    or meta.get("chunking_method") != method
                    or meta.get("embedding_revision") != self.active["embedding"]["revision"]
                    or meta.get("embedding_model") != self.active["embedding"]["model"]
                    or meta.get("document_id") not in eligible or meta.get("status") != "current"):
                raise RuntimeError("Pinecone returned incompatible source metadata")
            record_id, score = match["id"], float(match["score"])
            if not math.isfinite(score):
                raise RuntimeError("Pinecone returned an invalid similarity score")
            if record_id in seen:
                continue
            seen.add(record_id)
            section = "introduction" if meta["section"] == "intro" else f"section {meta['section']}"
            results.append({"chunk_id": record_id, "score": score,
                            "citation": f"[{meta['document_id']} v{meta['version']} {section}]",
                            "text": meta["text"], "metadata": meta})
        return sorted(results, key=lambda result: (-result["score"], result["chunk_id"]))[:top_k]

    def retrieve(self, question, methods=("fixed", "semantic"), top_k=5,
                 document_ids=None, as_of=None):
        vector = self.embed_question(question)
        return {method: self.query(vector, method, top_k, document_ids, as_of) for method in methods}


def print_matches(results, show_text=False):
    for method, matches in results.items():
        print(f"\n{method.upper()}: {len(matches)} matches (similarity scores, not confidence)")
        for rank, match in enumerate(matches, 1):
            print(f"{rank}. {match['score']:.4f} {match['citation']} {match['metadata']['title']}")
            text = match["text"] if show_text else match["text"][:280].replace("\n", " ")
            print(text)
