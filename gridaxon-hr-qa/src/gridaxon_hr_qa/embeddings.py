"""Embed complete chunks locally, using one retrieval model for both strategies."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from gridaxon_hr_qa.ingestion import write_documents


class ChunkEncoder:
    def __init__(self, revision="main"):
        from fastembed import TextEmbedding
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        model = "BAAI/bge-small-en-v1.5"
        repository = "Qdrant/bge-small-en-v1.5-onnx-Q"
        snapshot = snapshot_download(repository, revision=revision,
                                     allow_patterns=["*.json", "*.txt", "model_optimized.onnx"])
        self.model = TextEmbedding(model_name=model, specific_model_path=snapshot,
                                   local_files_only=True, threads=4,
                                   providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.identity = {"model": model, "repository": repository,
                         "revision": Path(snapshot).name, "backend": "fastembed-onnx",
                         "dimension": 384, "max_tokens": 512, "normalized": True,
                         "document_template": "title + section_title + text; v1",
                         "query_prefix": "Represent this sentence for searching relevant passages: "}

    def encode(self, texts):
        for index, text in enumerate(texts):
            count = len(self.tokenizer.encode(text).ids)
            if count > self.identity["max_tokens"]:
                raise ValueError(f"Embedding input {index} has {count} model tokens; "
                                 "reduce chunk size before embedding (no truncation allowed)")
        values = np.asarray(list(self.model.embed(texts, batch_size=32)), dtype=float)
        return validate_vectors(values, len(texts), self.identity["dimension"])

    def encode_query(self, question):
        return self.encode([self.identity["query_prefix"] + question])[0]


def validate_vectors(values, count, dimension):
    values = np.asarray(values, dtype=float)
    if values.shape != (count, dimension) or not np.isfinite(values).all():
        raise ValueError("Invalid embedding dimensions or non-finite vector values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero-length embedding vector")
    return values / norms


def embedding_text(chunk):
    return f"{chunk['metadata']['title']}\n{chunk['section_title']}\n\n{chunk['text']}"


def pinecone_metadata(chunk, identity, build_id):
    # Pinecone metadata is flat; preserve any complex YAML fields as JSON strings.
    metadata = {}
    for key, value in chunk["metadata"].items():
        if isinstance(value, (str, bool, int, float)):
            metadata[key] = value
        elif value is not None:
            metadata[key] = json.dumps(value, ensure_ascii=False)
    for key in ("section", "section_title", "source_path", "source_sha256",
                "start_char", "end_char", "section_start_char", "section_end_char",
                "chunk_index_in_section", "token_count", "text"):
        metadata[key] = chunk[key]
    metadata.update({"chunk_id": chunk["chunk_id"], "chunking_method": chunk["chunking"]["method"],
                     "chunking_config": json.dumps(chunk["chunking"], sort_keys=True),
                     "embedding_model": identity["model"], "embedding_revision": identity["revision"],
                     "build_id": build_id})
    if len(json.dumps(metadata, ensure_ascii=False).encode()) > 39000:
        raise ValueError(f"Metadata too large for chunk {chunk['chunk_id']}")
    return metadata


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def prepare_embeddings(storage: Path, encoder):
    inputs = {}
    for method in ("fixed", "semantic"):
        path = storage / f"chunks_{method}.jsonl"
        chunks = read_jsonl(path)
        if not chunks or len({c["chunk_id"] for c in chunks}) != len(chunks):
            raise ValueError(f"Empty or duplicate {method} chunks")
        if any(c["chunking"]["method"] != method for c in chunks):
            raise ValueError(f"Wrong chunking method in {path.name}")
        inputs[method] = chunks
    source_sets = [{(c["metadata"]["document_id"], c["source_sha256"])
                    for c in chunks} for chunks in inputs.values()]
    if source_sets[0] != source_sets[1]:
        raise ValueError("Fixed and semantic chunks must cover the same source versions")
    fingerprint = json.dumps([encoder.identity, inputs], sort_keys=True).encode()
    build_id = sha256(fingerprint).hexdigest()[:20]
    manifest = {"format": 1, "build_id": build_id, "embedding": encoder.identity,
                "metric": "cosine", "datasets": {}}
    # Validate and embed both strategies before publishing the manifest.
    prepared = {}
    for method, chunks in inputs.items():
        vectors = validate_vectors(encoder.encode([embedding_text(c) for c in chunks]),
                                   len(chunks), encoder.identity["dimension"])
        prepared[method] = [{"id": c["chunk_id"], "values": vector.tolist(),
                             "metadata": pinecone_metadata(c, encoder.identity, build_id)}
                            for c, vector in zip(chunks, vectors, strict=True)]
    for method, records in prepared.items():
        filename = f"vectors_{method}_{build_id}.jsonl"
        path = storage / filename
        write_documents(records, path)
        manifest["datasets"][method] = {
            "namespace": f"{method}-{build_id}", "file": filename, "count": len(records),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    write_json(storage / "embedding_manifest.json", manifest)
    return manifest


def load_build(storage):
    manifest = json.loads((storage / "embedding_manifest.json").read_text(encoding="utf-8"))
    if set(manifest["datasets"]) != {"fixed", "semantic"}:
        raise ValueError("Embedding build must contain fixed and semantic datasets")
    dimension = manifest["embedding"]["dimension"]
    datasets = {}
    for method, dataset in manifest["datasets"].items():
        path = storage / dataset["file"]
        if path.resolve().parent != storage.resolve():
            raise ValueError("Vector artifact must be inside storage")
        if sha256(path.read_bytes()).hexdigest() != dataset["sha256"]:
            raise ValueError(f"Checksum mismatch for {method}; regenerate embeddings")
        records = read_jsonl(path)
        if len(records) != dataset["count"] or not records:
            raise ValueError("Vector record count mismatch")
        if len({record["id"] for record in records}) != len(records):
            raise ValueError("Duplicate vector IDs")
        validate_vectors([record["values"] for record in records], len(records), dimension)
        if any(record["metadata"]["build_id"] != manifest["build_id"] for record in records):
            raise ValueError("Vector build identity mismatch")
        if dataset["namespace"] != f"{method}-{manifest['build_id']}":
            raise ValueError("Unexpected namespace for embedding build")
        datasets[method] = records
    return manifest, datasets
