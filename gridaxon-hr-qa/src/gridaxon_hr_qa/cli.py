"""Command-line entry point for the step-by-step RAG build."""

import argparse
from pathlib import Path

from gridaxon_hr_qa.ingestion import load_policies, write_documents


def main() -> None:
    parser = argparse.ArgumentParser(prog="gridaxon-hr-qa")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Read and validate current Markdown policies")
    ingest.add_argument("--data-dir", type=Path, required=True,
                        help="Starter folder containing corpus/")
    ingest.add_argument("--output", type=Path, default=Path("storage/documents.jsonl"))
    chunk = commands.add_parser("chunk-fixed", help="Split policy sections into fixed token windows")
    chunk.add_argument("--input", type=Path, default=Path("storage/documents.jsonl"))
    chunk.add_argument("--output", type=Path, default=Path("storage/chunks_fixed.jsonl"))
    chunk.add_argument("--chunk-size", type=int, default=300)
    chunk.add_argument("--overlap", type=int, default=50)
    semantic = commands.add_parser("chunk-semantic", help="Group sentences by local embedding similarity")
    semantic.add_argument("--input", type=Path, default=Path("storage/documents.jsonl"))
    semantic.add_argument("--output", type=Path, default=Path("storage/chunks_semantic.jsonl"))
    semantic.add_argument("--min-tokens", type=int, default=80)
    semantic.add_argument("--max-tokens", type=int, default=300)
    semantic.add_argument("--distance-threshold", type=float, default=0.45)
    semantic.add_argument("--model-revision", default="main",
                          help="Model commit or revision; resolved commit is saved in chunk metadata")
    embed = commands.add_parser("embed-chunks", help="Embed both chunk sets locally")
    embed.add_argument("--storage-dir", type=Path, default=Path("storage"))
    embed.add_argument("--model-revision", default="main")
    index = commands.add_parser("index-chunks", help="Upload and verify both Pinecone namespaces")
    index.add_argument("--storage-dir", type=Path, default=Path("storage"))
    index.add_argument("--env-file", type=Path, default=Path(".env"))
    index.add_argument("--create-index", action="store_true")
    for name, help_text in (("retrieve", "Search Pinecone directly"),
                            ("route-query", "Route and retrieve using LangGraph")):
        search = commands.add_parser(name, help=help_text)
        search.add_argument("question")
        search.add_argument("--method", choices=("fixed", "semantic", "both"), default="both")
        search.add_argument("--top-k", type=int, default=5)
        search.add_argument("--as-of", help="ISO date; current index has no archived versions")
        search.add_argument("--storage-dir", type=Path, default=Path("storage"))
        search.add_argument("--env-file", type=Path, default=Path(".env"))
        search.add_argument("--output", type=Path, help="Optional JSON result file")
        search.add_argument("--show-text", action="store_true", help="Print full source passages")
        if name == "retrieve":
            search.add_argument("--document-id", action="append", help="Restrict to policy ID; repeat for multiple policies")
    check = commands.add_parser("check-retrieval", help="Run four live development retrieval checks")
    check.add_argument("--top-k", type=int, default=5)
    check.add_argument("--storage-dir", type=Path, default=Path("storage"))
    check.add_argument("--env-file", type=Path, default=Path(".env"))
    check.add_argument("--output", type=Path, default=Path("storage/retrieval_checks.json"))
    grade = commands.add_parser("grade-query", help="Retrieve and grade evidence before answer generation")
    grade.add_argument("question")
    grade.add_argument("--method", choices=("fixed", "semantic", "both"), default="both")
    grade.add_argument("--top-k", type=int, default=5)
    grade.add_argument("--as-of")
    grade.add_argument("--model", help="Override OPENAI_GRADER_MODEL")
    grade.add_argument("--storage-dir", type=Path, default=Path("storage"))
    grade.add_argument("--env-file", type=Path, default=Path(".env"))
    grade.add_argument("--output", type=Path, default=Path("storage/graded_query.json"))
    answer = commands.add_parser("answer-query", help="Generate and validate cited answers from approved evidence")
    answer.add_argument("question")
    answer.add_argument("--method", choices=("fixed", "semantic", "both"), default="both")
    answer.add_argument("--top-k", type=int, default=5)
    answer.add_argument("--as-of")
    answer.add_argument("--model", help="Model for grading, generation, and validation")
    answer.add_argument("--storage-dir", type=Path, default=Path("storage"))
    answer.add_argument("--env-file", type=Path, default=Path(".env"))
    answer.add_argument("--output", type=Path, default=Path("storage/answered_query.json"))
    args = parser.parse_args()
    if args.command in ("grade-query", "answer-query"):
        from gridaxon_hr_qa.embeddings import write_json
        from gridaxon_hr_qa.retrieval import Retriever
        from gridaxon_hr_qa.grading import EvidenceGrader
        from gridaxon_hr_qa.graded_workflow import run_graded
        from gridaxon_hr_qa.openai_grader import OpenAIGraderBackend

        try:
            if not 1 <= args.top_k <= 20:
                raise ValueError("For grading, top-k must be between 1 and 20")
            backend = OpenAIGraderBackend.from_env(args.env_file, args.model)
            retriever = Retriever.from_storage(args.storage_dir, args.env_file)
            methods = ("fixed", "semantic") if args.method == "both" else (args.method,)
            if args.command == "answer-query":
                from gridaxon_hr_qa.answering import AnswerEngine
                from gridaxon_hr_qa.answer_workflow import run_answered

                print(f"Retrieving, grading, and validating answers with {backend.model}...", flush=True)
                result = run_answered(retriever, EvidenceGrader(backend), AnswerEngine(backend),
                                      args.question, methods, args.top_k, args.as_of)
            else:
                print(f"Grading with {backend.model}; no answer will be generated.", flush=True)
                result = run_graded(retriever, EvidenceGrader(backend), args.question, methods, args.top_k, args.as_of)
            result["llm_calls"] = backend.calls
            write_json(args.output, result)
            print("Graph: " + " -> ".join(result["trace"]))
            if args.command == "answer-query":
                for method, item in result["answers"].items():
                    print(f"\n{method.upper()} — {item['status']}")
                    print(item["text"])
                    print(f"Referenced source spans: {len(item['sources'])}")
                print(f"Saved {args.output.resolve()}")
                return
            for method, assessment in result["grades"].items():
                print(f"{method}: {assessment['decision']}; relevant {assessment['relevant_count']}/"
                      f"{assessment['retrieved_count']}; can_generate={assessment['can_generate']}")
                for requirement in assessment["coverage"]:
                    print(f"  {requirement['requirement_id']}: {requirement['status']} - {requirement['reason']}")
                if assessment["clarification_question"]:
                    print(assessment["clarification_question"])
            print(f"Saved {args.output.resolve()}")
        except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
            # Overwrite an earlier report with a blocked result, never leave a stale ready flag.
            write_json(args.output, {"question": args.question, "status": "error", "can_generate": False,
                                     "error": str(exc)})
            parser.exit(1, f"{args.command} blocked: {exc}\n")
        return
    if args.command in ("retrieve", "route-query", "check-retrieval"):
        from gridaxon_hr_qa.embeddings import write_json
        from gridaxon_hr_qa.retrieval import Retriever, print_matches

        try:
            if not 1 <= args.top_k <= 50:
                raise ValueError("top-k must be between 1 and 50")
            retriever = Retriever.from_storage(args.storage_dir, args.env_file)
            if args.command == "check-retrieval":
                from gridaxon_hr_qa.retrieval_checks import check_retrieval

                report = check_retrieval(retriever, args.top_k)
                write_json(args.output, report)
                print(f"{report['passed']}/{report['total']} evidence-section checks passed.")
                print(f"Exact reference sections: {report['strict_passed']}/{report['total']} "
                      "(accepted source alternatives are listed in the report).")
                print(f"Saved {args.output.resolve()}")
                if report["passed"] != report["total"]:
                    parser.exit(1, "Inspect the report for missed evidence; this is a development check.\n")
                return
            methods = ("fixed", "semantic") if args.method == "both" else (args.method,)
            if args.command == "route-query":
                from gridaxon_hr_qa.routing import run_routed

                result = run_routed(retriever, args.question, methods, args.top_k, args.as_of)
                print(f"Route: {result['route']['mode']} | {', '.join(result['route']['document_ids'])}")
                print(result["route"]["reason"])
                print("Graph: " + " -> ".join(result["trace"]))
                for warning in result["warnings"]:
                    print(warning)
            else:
                result = {"question": args.question, "top_k": args.top_k,
                          "results": retriever.retrieve(args.question, methods, args.top_k,
                                                         args.document_id, args.as_of)}
                print("Current-version retrieval candidates only; not a graded or generated answer.")
            print_matches(result["results"], args.show_text)
            if args.output:
                write_json(args.output, result)
        except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
            parser.exit(1, f"{args.command} failed: {exc}\n")
        return
    if args.command in ("embed-chunks", "index-chunks"):
        try:
            if args.command == "embed-chunks":
                from gridaxon_hr_qa.embeddings import ChunkEncoder, prepare_embeddings

                print("Loading local BGE chunk embedding model...", flush=True)
                manifest = prepare_embeddings(args.storage_dir, ChunkEncoder(args.model_revision))
            else:
                from gridaxon_hr_qa.pinecone_store import settings, upload_build

                manifest = upload_build(args.storage_dir, settings(args.env_file), args.create_index)
            for method, dataset in manifest["datasets"].items():
                print(f"{method}: {dataset['count']} vectors; namespace={dataset['namespace']}")
            print(f"Embedding dimensions: {manifest['embedding']['dimension']}")
        except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
            parser.exit(1, f"{args.command} failed: {exc}\n")
        return
    if args.command == "chunk-semantic":
        from collections import Counter
        from gridaxon_hr_qa.chunking import read_documents
        from gridaxon_hr_qa.semantic_chunking import (
            LocalSentenceEncoder, semantic_chunks, validate_settings,
        )

        try:
            if args.input.resolve() == args.output.resolve():
                raise ValueError("Input and output paths must be different")
            validate_settings(args.min_tokens, args.max_tokens, args.distance_threshold)
            documents = read_documents(args.input)
            print("Loading local sentence model (first run downloads model weights)...", flush=True)
            backend = LocalSentenceEncoder(revision=args.model_revision)
            chunks = semantic_chunks(documents, backend, args.min_tokens,
                                     args.max_tokens, args.distance_threshold)
            write_documents(chunks, args.output)
        except (ValueError, OSError, RuntimeError, ImportError) as exc:
            parser.exit(1, f"Semantic chunking failed: {exc}\n")
        print(f"Created {len(chunks)} semantic chunks from {len(documents)} policies.")
        print(f"Boundary reasons: {dict(Counter(c['boundary_reason'] for c in chunks))}")
        print(f"Model revision: {backend.identity['revision']}")
        print(f"Saved to {args.output.resolve()}")
        return
    if args.command == "chunk-fixed":
        from gridaxon_hr_qa.chunking import fixed_chunks, read_documents

        try:
            if args.input.resolve() == args.output.resolve():
                raise ValueError("Input and output paths must be different")
            documents = read_documents(args.input)
            chunks = fixed_chunks(documents, args.chunk_size, args.overlap)
            write_documents(chunks, args.output)
        except (ValueError, OSError) as exc:
            parser.exit(1, f"Chunking failed: {exc}\n")
        print(f"Created {len(chunks)} fixed chunks from {len(documents)} policies.")
        print(f"Token limit: {args.chunk_size}; requested overlap: {args.overlap}.")
        print(f"Saved to {args.output.resolve()}")
        return
    try:
        documents = load_policies(args.data_dir)
        write_documents(documents, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Ingestion failed: {exc}\n")
    for document in documents:
        meta = document["metadata"]
        print(f"{meta['document_id']} | v{meta['version']} | "
              f"{meta['effective_date']} | {meta['title']}")
    print(f"\nLoaded {len(documents)} current policies.")
    print(f"Saved to {args.output.resolve()}")
