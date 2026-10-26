"""Read current Markdown policies without ingesting archives or evaluation data."""

from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
import json

import frontmatter


REQUIRED_FIELDS = ("document_id", "title", "version", "effective_date", "status")


def json_safe(value):
    """YAML dates need converting before writing JSON."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def load_policies(data_dir: Path) -> list[dict]:
    """data_dir is the starter folder containing corpus/, not corpus itself."""
    corpus_dir = data_dir / "corpus"
    if not corpus_dir.is_dir():
        raise ValueError(f"Missing corpus folder: {corpus_dir}")
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown policies found in {corpus_dir}")

    documents = []
    seen_ids = set()
    for path in paths:
        raw_bytes = path.read_bytes()
        try:
            raw_text = raw_bytes.decode("utf-8-sig")
            policy = frontmatter.loads(raw_text)
            metadata = json_safe(policy.metadata)
            missing = [key for key in REQUIRED_FIELDS if not metadata.get(key)]
            if missing:
                raise ValueError(f"Missing metadata: {', '.join(missing)}")
            for key in ("document_id", "title", "version", "status"):
                metadata[key] = str(metadata[key])
            date.fromisoformat(str(metadata["effective_date"]))
            if metadata["status"] != "current":
                raise ValueError("Only current policies are allowed in corpus/")
            if not policy.content.strip():
                raise ValueError("Policy body is empty")
            document_id = metadata["document_id"]
            if document_id in seen_ids:
                raise ValueError(f"Duplicate current document ID: {document_id}")
            seen_ids.add(document_id)
        except Exception as exc:
            raise ValueError(f"Invalid policy {path.name}: {exc}") from exc

        documents.append({
            "metadata": metadata,
            "source_path": path.relative_to(data_dir).as_posix(),
            "source_sha256": sha256(raw_bytes).hexdigest(),
            "text": policy.content,
            "raw_markdown": raw_text,
        })
    return documents


def write_documents(documents: list[dict], output: Path) -> None:
    """Write only after the entire input has passed validation."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    temporary.replace(output)
