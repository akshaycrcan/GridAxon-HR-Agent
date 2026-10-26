"""GridAxon People Hub — run with uv run streamlit run streamlit_app.py."""

import json
import streamlit as st

from gridaxon_hr_qa.portal import (
    STORAGE, SECRET_NAMES, ask_policy, citation_groups, highlighted_document,
    linked_answer, markdown_text,
)

st.set_page_config(page_title="GridAxon HR Q&A", page_icon=":material/hub:", layout="wide")

for key, default in {"question": "", "answer": None, "answered_question": "", "error": None}.items():
    st.session_state.setdefault(key, default)


@st.cache_data(max_entries=2, show_spinner=False)
def load_library(signature):
    return [json.loads(line) for line in (STORAGE / "documents.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def suggest(question):
    st.session_state.question = question


@st.dialog("Policy document", width="large")
def view_document(document, sources):
    meta = document["metadata"]
    st.subheader(meta["title"])
    st.caption(f"{meta['document_id']} · Version {meta['version']} · Effective {meta['effective_date']}")
    if sources:
        st.caption("Referenced passages are highlighted in yellow in the policy body below.")
    with st.container(height=520, border=True):
        # Native Markdown cannot highlight exact character ranges. All document
        # content is escaped before adding the app-owned mark tags.
        st.html(highlighted_document(document, sources))
    st.download_button("Download Markdown", document["raw_markdown"],
                       file_name=f"{meta['document_id']}.md", mime="text/markdown", icon=":material/download:")


with st.container(horizontal=True, horizontal_alignment="distribute"):
    st.markdown("### :orange[:material/hub:] GridAxon  ·  People Hub")
    st.badge("Employee portal", color="blue", icon=":material/apartment:")
st.caption("People Hub / HR policies")
st.title("HR policies, explained.")
st.markdown("Ask a question or browse your company policies.")
st.badge("Demo policy corpus", color="gray", icon=":material/description:")

try:
    documents = load_library((STORAGE / "documents.jsonl").stat().st_mtime_ns)
except (OSError, ValueError):
    st.error("The policy library is unavailable. Run the Markdown ingestion step before opening the portal.")
    st.stop()
by_id = {doc["metadata"]["document_id"]: doc for doc in documents}
library, workspace = st.columns([1, 2.4], gap="medium")
library_slot = library.container(border=True)

with workspace:
    with st.container(border=True):
        st.subheader("Ask an HR policy question")
        st.caption("Get an answer with references to the policy documents. Each question is answered independently.")
        with st.form("ask", border=False):
            question_col, submit_col = st.columns([4, 1], vertical_alignment="bottom")
            question_col.text_input("Your question", key="question", max_chars=2000,
                                    placeholder="How much PTO do full-time employees accrue?", label_visibility="collapsed")
            submitted = submit_col.form_submit_button("Ask question", type="primary", width="stretch")
        with st.container(horizontal=True):
            for title, question in [("PTO & leave", "How much PTO do full-time employees accrue?"),
                                    ("Remote work", "What is the remote and hybrid work policy?"),
                                    ("Travel expenses", "How do I get business travel expenses reimbursed?")]:
                st.button(title, on_click=suggest, args=(question,), key=f"suggest_{title}")

    if submitted:
        st.session_state.answer = None
        st.session_state.error = None
        st.session_state.answered_question = ""
        question = st.session_state.question.strip()
        if not question or len(question) > 2000:
            st.session_state.error = "Enter a policy question between 1 and 2,000 characters."
        else:
            with st.status("Preparing your question…", expanded=False) as status:
                try:
                    try:
                        secrets = {key: st.secrets[key] for key in SECRET_NAMES if key in st.secrets}
                    except FileNotFoundError:
                        secrets = {}
                    st.session_state.answer = ask_policy(question, lambda label: status.update(label=label), secrets)
                    st.session_state.answered_question = question
                    status.update(label="Policy review complete", state="complete")
                except Exception:
                    # Never send provider responses, credentials, local paths, or
                    # partially checked text to an employee browser.
                    st.session_state.error = "We couldn't complete the policy check. Please try again. If it continues, ask the app administrator to check API access and the indexed policy files."
                    status.update(label="Policy check unavailable", state="error")

    if st.session_state.error:
        st.error(st.session_state.error)
    answer = st.session_state.answer
    if answer:
        with st.container(border=True):
            st.subheader("Your answer")
            st.caption(f"Question: {st.session_state.answered_question}")
            if answer["status"] == "answered":
                groups = citation_groups(answer)
                count = len({g["metadata"]["document_id"] for g in groups.values()})
                st.badge(f"Based on {count} {'policy' if count == 1 else 'policies'}", color="blue")
                st.markdown(linked_answer(answer))
                st.subheader("Referenced policies")
                for label, group in groups.items():
                    meta, number = group["metadata"], group["number"]
                    with st.container(border=True):
                        st.subheader(f"[{number}] {meta['title']}", anchor=f"source-{number}")
                        st.caption(f"{meta['document_id']} · Version {meta['version']} · Section {meta['section']}")
                        with st.expander("View cited excerpt", expanded=True):
                            for source in group["spans"]:
                                st.info(markdown_text(source["text"]), icon=":material/format_quote:")
                        if meta["document_id"] in by_id and st.button("Open full document", key=f"source_{number}", icon=":material/open_in_new:"):
                            view_document(by_id[meta["document_id"]], group["spans"])
            else:
                st.info(markdown_text(answer["text"]), icon=":material/help:")
            st.caption("Automated answers can be imperfect. Check the cited policy. For personal cases, contact People Operations.")
    elif not st.session_state.error:
        with st.container(border=True):
            st.subheader("A clear answer. A policy you can check.", icon=":material/forum:")
            st.markdown("Ask about leave, working arrangements, expenses, or another HR policy. Referenced documents will be highlighted in the library.")
            st.caption("No personal employee records are connected to this app.")

with library_slot:
    st.subheader("Policy library", icon=":material/folder_open:")
    st.caption(f"{len(documents)} current documents")
    search = st.text_input("Find a policy", placeholder="Search title or policy ID…", key="policy_search")
    answer = st.session_state.answer
    groups = citation_groups(answer) if answer and answer["status"] == "answered" else {}
    referenced = {}
    for group in groups.values():
        referenced.setdefault(group["metadata"]["document_id"], []).append(group["number"])
    filtered = [doc for doc in documents if search.casefold() in
                (doc["metadata"]["title"] + " " + doc["metadata"]["document_id"]).casefold()]
    filtered.sort(key=lambda doc: (doc["metadata"]["document_id"] not in referenced, doc["metadata"]["document_id"]))
    with st.container(height=580, border=False):
        if not filtered:
            st.caption("No policies match. Try a different title or ID.")
        for document in filtered:
            meta = document["metadata"]
            identifier = meta["document_id"]
            with st.container(border=True, gap="xsmall"):
                if identifier in referenced:
                    st.badge("Referenced in this answer · " + ", ".join(f"[{n}]" for n in referenced[identifier]), color="blue")
                if st.button(meta["title"], key=f"doc_{identifier}", width="stretch", wrap=True,
                             type="primary" if identifier in referenced else "tertiary", icon=":material/description:"):
                    sources = [s for s in (answer or {}).get("sources", []) if s["metadata"]["document_id"] == identifier]
                    view_document(document, sources)
                st.caption(f"{identifier} · v{meta['version']}")
    st.caption(f"Showing {len(filtered)} of {len(documents)} policies")
