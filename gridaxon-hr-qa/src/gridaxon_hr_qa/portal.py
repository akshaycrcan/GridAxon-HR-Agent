"""Employee-facing projection and service adapter; never expose rejected drafts."""

from html import escape
from pathlib import Path
import re
from threading import RLock

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
STORAGE = ROOT / "storage"
SECRET_NAMES = {"OPENAI_API_KEY", "OPENAI_GRADER_MODEL", "PINECONE_API_KEY", "PINECONE_INDEX"}


def public_answer(result, method="fixed"):
    answer = result["answers"][method]
    if answer.get("status") != "answered" or answer.get("validated") is not True:
        messages = {
            "clarify": answer.get("text", "Please provide more detail about your question."),
            "resolve_conflict": "The policies contain conflicting evidence. Please ask People Operations to review your question.",
            "abstain": "I couldn't find and validate enough policy evidence to answer. Try a more specific question or contact People Operations.",
        }
        return {"status": answer.get("status", "abstain"),
                "text": messages.get(answer.get("status"), messages["abstain"]), "sources": []}
    if not result.get("checks", {}).get(method, {}).get("passed"):
        raise ValueError("Answer validation is missing")
    # Only the published answer crosses the UI boundary, never drafts or attempts.
    return {"status": "answered", "text": answer["text"], "sources": answer["sources"]}


def citation_groups(answer):
    groups = {}
    for source in answer.get("sources", []):
        label = source["citation"]
        if label not in groups:
            groups[label] = {"number": len(groups) + 1, "metadata": source["metadata"], "spans": []}
        groups[label]["spans"].append(source)
    return groups


def markdown_text(text):
    """Display generated text as text, allowing only app-created citation links."""
    return re.sub(r"([\\`*_{}\[\]()<>#+.!|~\-])", r"\\\1", text)


def linked_answer(answer):
    groups = citation_groups(answer)
    pattern = "(" + "|".join(re.escape(label) for label in sorted(groups, key=len, reverse=True)) + ")"
    if not groups:
        return markdown_text(answer["text"])
    return "".join(f"[[{groups[part]['number']}]](#source-{groups[part]['number']})"
                   if part in groups else markdown_text(part)
                   for part in re.split(pattern, answer["text"]))


def highlighted_document(document, sources):
    """Escape all policy HTML and highlight only source-verified body offsets."""
    text = document["text"]
    ranges = []
    for source in sources:
        meta = source["metadata"]
        if (meta.get("document_id") != document["metadata"]["document_id"]
                or meta.get("source_sha256") != document["source_sha256"]):
            continue
        start = meta["start_char"] + source["quote_start_char"]
        end = meta["start_char"] + source["quote_end_char"]
        if 0 <= start < end <= len(text) and text[start:end] == source["text"]:
            ranges.append((start, end))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    parts, position = [], 0
    for start, end in merged:
        parts.extend([escape(text[position:start]), "<mark>", escape(text[start:end]), "</mark>"])
        position = end
    parts.append(escape(text[position:]))
    return '<div style="white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.7">' + "".join(parts) + "</div>"


@st.cache_resource(max_entries=2, show_spinner=False)
def encoder_resource(revision):
    from gridaxon_hr_qa.embeddings import ChunkEncoder
    return ChunkEncoder(revision=revision), RLock()


def ask_policy(question, progress, secrets=None):
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("Enter a question between 1 and 2,000 characters.")
    from gridaxon_hr_qa.answer_workflow import build_answer_graph
    from gridaxon_hr_qa.answering import AnswerEngine
    from gridaxon_hr_qa.grading import EvidenceGrader
    from gridaxon_hr_qa.openai_grader import OpenAIGraderBackend
    from gridaxon_hr_qa.pinecone_store import PineconeREST, read_env_values
    from gridaxon_hr_qa.retrieval import Retriever, load_active

    config = read_env_values(ROOT / ".env", SECRET_NAMES)
    config.update({k: v for k, v in (secrets or {}).items() if k in SECRET_NAMES and v})
    backend = OpenAIGraderBackend(config.get("OPENAI_API_KEY"), config.get("OPENAI_GRADER_MODEL") or "gpt-5-mini")
    if not config.get("PINECONE_API_KEY"):
        raise ValueError("Pinecone credentials are missing")
    active, datasets = load_active(STORAGE)
    if config.get("PINECONE_INDEX", "gridaxon-hr-qa") != active["index"]:
        raise ValueError("Configured index differs from the verified build")
    progress("Loading the policy search model…")
    encoder, lock = encoder_resource(active["embedding"]["revision"])
    retriever = Retriever(active, datasets, encoder, PineconeREST(config["PINECONE_API_KEY"]))
    graph = build_answer_graph(retriever, EvidenceGrader(backend), AnswerEngine(backend))
    labels = {"grade_evidence": "Writing an answer from approved evidence…",
              "generate_answers": "Checking answer support and citations…",
              "check_groundedness_and_citations": "Finalizing the answer…",
              "revise_once": "Checking the revised answer…",
              "recheck_revision": "Finalizing the answer…"}
    state = {}
    # Protect the shared local model; per-question clients and outputs are not cached.
    with lock:
        progress("Finding policies and checking evidence coverage…")
        for update in graph.stream({"question": question.strip(), "methods": ["fixed"], "top_k": 5},
                                   config={"recursion_limit": 10}, stream_mode="updates"):
            for node, values in update.items():
                state.update(values)
                if node in labels:
                    progress(labels[node])
    return public_answer(state)
