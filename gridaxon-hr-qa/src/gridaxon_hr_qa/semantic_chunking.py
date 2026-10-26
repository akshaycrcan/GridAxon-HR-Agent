"""Group adjacent sentences by embedding similarity within policy sections."""

from copy import deepcopy
from hashlib import sha256
import json
import math

import numpy as np
import pysbd
import tiktoken

from gridaxon_hr_qa.chunking import sections


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalSentenceEncoder:
    """Download public model weights, then embed policy sentences on the CPU."""

    def __init__(self, model_name=DEFAULT_MODEL, revision="main"):
        from pathlib import Path
        from huggingface_hub import snapshot_download
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer

        repository = "Qdrant/all-MiniLM-L6-v2-onnx"
        snapshot = snapshot_download(
            repository, revision=revision,
            allow_patterns=["*.json", "model.onnx", "*.txt"],
        )
        self.model = TextEmbedding(model_name=model_name, specific_model_path=snapshot,
                                   local_files_only=True, threads=4,
                                   providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.identity = {"model": model_name, "revision": Path(snapshot).name,
                         "repository": repository, "backend": "fastembed-onnx",
                         "max_sequence_length": 256}

    def fits(self, text):
        return len(self.tokenizer.encode(text).ids) <= self.identity["max_sequence_length"]

    def encode(self, texts):
        return np.asarray(list(self.model.embed(texts, batch_size=32)))


def sentence_spans(text):
    """Retain all whitespace and attach the section heading to its first sentence."""
    heading_end = text.find("\n") + 1 if text.lstrip().startswith("#") else 0
    body = text[heading_end:]
    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    sentences = segmenter.segment(body)
    ends = [heading_end + sentence.end for sentence in sentences[:-1]]
    start = 0
    for end in [*ends, len(text)]:
        if end > start:
            yield start, end
            start = end


def bounded_units(text, tokenizer, backend, max_tokens):
    """Split unusually long sentences before embedding; never silently truncate."""
    def split(start, end):
        part = text[start:end]
        if (len(tokenizer.encode(part, disallowed_special=())) <= max_tokens
                and backend.fits(part)):
            return [(start, end)]
        if end - start <= 1:
            raise ValueError("Token limits are too small for a Unicode character")
        midpoint = (start + end) // 2
        # Prefer a word boundary near the middle, preserving exact offsets.
        left_space = text.rfind(" ", start + 1, midpoint + 1)
        cut = left_space + 1 if left_space > start else midpoint
        return split(start, cut) + split(cut, end)

    return [unit for start, end in sentence_spans(text) for unit in split(start, end)]


def validate_settings(min_tokens, max_tokens, distance_threshold):
    if not 0 < min_tokens <= max_tokens:
        raise ValueError("Require 0 < min-tokens <= max-tokens")
    if not math.isfinite(distance_threshold) or not 0 <= distance_threshold <= 2:
        raise ValueError("distance-threshold must be between 0 and 2")


def semantic_chunks(documents, backend, min_tokens=80, max_tokens=300,
                    distance_threshold=0.45, encoding="cl100k_base"):
    """A larger cosine-distance threshold produces fewer semantic breaks.

    min_tokens is a soft floor for semantic breaks; max_tokens is a hard ceiling.
    Short sections/tails are retained. These chunks do not overlap.
    """
    validate_settings(min_tokens, max_tokens, distance_threshold)
    tokenizer = tiktoken.get_encoding(encoding)
    config = {"method": "semantic", "implementation": "1", "encoding": encoding,
              "min_tokens": min_tokens, "max_tokens": max_tokens, "overlap": 0,
              "distance_threshold": distance_threshold,
              "sentence_splitter": "pysbd-en", "embedding": backend.identity}
    prepared, texts = [], []
    for document in documents:
        body = document["text"]
        if not isinstance(body, str) or not body.strip():
            raise ValueError("Each document must have a non-empty text body")
        for reference, title, start, end in sections(body):
            section_text = body[start:end]
            units = bounded_units(section_text, tokenizer, backend, max_tokens)
            prepared.append((document, reference, title, start, end, units, len(texts)))
            texts.extend(section_text[a:b] for a, b in units)
    if not texts:
        return []
    vectors = np.asarray(backend.encode(texts), dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("Embedding backend returned invalid vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Embedding backend returned zero-length vectors")
    vectors = vectors / norms
    chunks = []
    for document, reference, title, section_start, section_end, units, offset in prepared:
        body = document["text"]
        metadata = document["metadata"]
        first, ordinal = 0, 0
        for index, (_, end) in enumerate(units):
            start = units[first][0]
            text = body[section_start + start:section_start + end]
            count = len(tokenizer.encode(text, disallowed_special=()))
            last = index == len(units) - 1
            distance = None
            reason = None
            if last:
                reason = "section_end"
            else:
                distance = float(np.clip(1 - np.dot(vectors[offset + index],
                                                   vectors[offset + index + 1]), 0, 2))
                extended = body[section_start + start:section_start + units[index + 1][1]]
                if len(tokenizer.encode(extended, disallowed_special=())) > max_tokens:
                    reason = "max_tokens"
                elif count >= min_tokens and distance >= distance_threshold:
                    remaining = body[section_start + end:section_end]
                    remaining_count = len(tokenizer.encode(remaining, disallowed_special=()))
                    if remaining_count >= min_tokens:
                        reason = "semantic_break"
            if reason is None:
                continue
            absolute_start, absolute_end = section_start + start, section_start + end
            identity = json.dumps([document["source_sha256"], config, metadata,
                                   absolute_start, absolute_end], sort_keys=True)
            digest = sha256(identity.encode()).hexdigest()
            chunks.append({
                "chunk_id": f"{metadata['document_id']}_v{metadata['version']}_semantic_{digest}",
                "metadata": deepcopy(metadata),
                "source_path": document["source_path"],
                "source_sha256": document["source_sha256"],
                "section": reference, "section_title": title,
                "section_start_char": section_start, "section_end_char": section_end,
                "start_char": absolute_start, "end_char": absolute_end,
                "chunk_index_in_section": ordinal,
                "chunking": deepcopy(config), "token_count": count, "text": text,
                "boundary_reason": reason, "distance_to_next_unit": distance,
            })
            first = index + 1
            ordinal += 1
    return chunks
