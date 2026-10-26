"""Fixed token windows confined to Markdown H2 sections."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

import tiktoken


def sections(text: str):
    """Yield (reference, title, start, end); offsets index the ingested body."""
    headings = []
    position = 0
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            run = marker.group(1)
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence):
                fence = None
        elif fence is None:
            heading = re.match(r"^ {0,3}##[ \t]+(.+?)\s*$", line)
            if heading:
                title = re.sub(r"\s+#+$", "", heading.group(1))
                number = re.match(r"(\d+)\.[ \t]+", title)
                reference = number.group(1) if number else f"heading-{len(headings) + 1}"
                headings.append((reference, title, position))
        position += len(line)
    if not headings or headings[0][2] > 0:
        headings.insert(0, ("intro", "Introduction", 0))
    for index, (reference, title, start) in enumerate(headings):
        end = headings[index + 1][2] if index + 1 < len(headings) else len(text)
        if text[start:end].strip():
            yield reference, title, start, end


def token_windows(text, encoder, size, overlap):
    """Preserve exact Unicode text; adjust boundaries to whole characters."""
    tokens = encoder.encode(text, disallowed_special=())
    byte_to_char = {0: 0}
    byte_position = 0
    for index, char in enumerate(text):
        byte_position += len(char.encode("utf-8"))
        byte_to_char[byte_position] = index + 1
    boundaries = [0]
    for token in tokens:
        boundaries.append(boundaries[-1] + len(encoder.decode_single_token_bytes(token)))
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        while end > start:
            if boundaries[end] in byte_to_char:
                char_start = byte_to_char[boundaries[start]]
                char_end = byte_to_char[boundaries[end]]
                count = len(encoder.encode(text[char_start:char_end], disallowed_special=()))
                if count <= size:
                    break
            end -= 1
        if end == start:
            raise ValueError("Chunk size is too small to preserve a Unicode character")
        yield char_start, char_end, count
        if end == len(tokens):
            break
        next_start = max(start + 1, end - overlap)
        while boundaries[next_start] not in byte_to_char:
            next_start += 1
        start = next_start


def fixed_chunks(documents: list[dict], chunk_size=300, overlap=50,
                 encoding="cl100k_base") -> list[dict]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk-size > 0 and 0 <= overlap < chunk-size")
    encoder = tiktoken.get_encoding(encoding)
    config = {"method": "fixed", "implementation": "1", "encoding": encoding,
              "chunk_size": chunk_size, "overlap": overlap}
    chunks = []
    for document in documents:
        text = document["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Each document must have a non-empty text body")
        metadata = document["metadata"]
        for reference, title, section_start, section_end in sections(text):
            section_text = text[section_start:section_end]
            for ordinal, (start, end, count) in enumerate(
                    token_windows(section_text, encoder, chunk_size, overlap)):
                start += section_start
                end += section_start
                identity = json.dumps([document["source_sha256"], config,
                                       metadata, start, end], sort_keys=True)
                digest = sha256(identity.encode()).hexdigest()
                chunks.append({
                    "chunk_id": f"{metadata['document_id']}_v{metadata['version']}_fixed_{digest}",
                    "metadata": deepcopy(metadata),
                    "source_path": document["source_path"],
                    "source_sha256": document["source_sha256"],
                    "section": reference,
                    "section_title": title,
                    "section_start_char": section_start,
                    "section_end_char": section_end,
                    "start_char": start,
                    "end_char": end,
                    "chunk_index_in_section": ordinal,
                    "chunking": dict(config),
                    "token_count": count,
                    "text": text[start:end],
                })
    return chunks


def read_documents(path: Path) -> list[dict]:
    documents = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
                for key in ("metadata", "text", "source_path", "source_sha256"):
                    if key not in document:
                        raise ValueError(f"Missing {key}")
                if not isinstance(document["metadata"], dict):
                    raise ValueError("metadata must be an object")
                for key in ("document_id", "version"):
                    if not document["metadata"].get(key):
                        raise ValueError(f"Missing metadata.{key}")
                documents.append(document)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid document on line {line_number}: {exc}") from exc
    if not documents:
        raise ValueError("Input contains no documents; run ingestion first")
    return documents
