"""Small live development checks, not the held-out RAG evaluation benchmark."""

from time import perf_counter

from gridaxon_hr_qa.routing import run_routed


CASES = [
    ("pto", "How much PTO do regular full-time employees accrue per year?", [("HR-008", "2.0", "1")]),
    ("complaint", "Who can I report harassment to if the CEO is involved?", [("HR-001", "1.0", "2")]),
    ("remote", "Who pays for home internet and equipment for remote work?", [("HR-011", "1.0", "3")]),
    ("multi_policy", "Does signing the handbook execute a non-compete?",
     [("HR-006", "1.0", "2"), ("HR-005", "1.0", "1")]),
]

# Manually checked against the source: HR-006 section 4 explicitly states that
# acknowledging the handbook does not execute a restrictive covenant. The supplied
# seed label cites section 2, which supports the same claim. Retain strict misses.
ALTERNATIVES = {("HR-006", "1.0", "2"): [("HR-006", "1.0", "4")]}


def score_evidence(expected, matches):
    found = {(m["metadata"]["document_id"], m["metadata"]["version"],
              m["metadata"]["section"]) for m in matches}
    strict_missing = [list(item) for item in expected if tuple(item) not in found]
    missing, alternatives_used = [], []
    for item in expected:
        key = tuple(item)
        if key in found:
            continue
        alternatives = [candidate for candidate in ALTERNATIVES.get(key, []) if candidate in found]
        if alternatives:
            alternatives_used.append({"reference": list(key), "accepted": list(alternatives[0])})
        else:
            missing.append(list(item))
    return {"missing_sections": missing, "strict_missing_sections": strict_missing,
            "accepted_alternatives": alternatives_used, "all_sections_found": not missing,
            "all_exact_sections_found": not strict_missing}


def check_retrieval(retriever, top_k=5):
    report = {"top_k": top_k, "as_of": "2026-10-24", "build_id": retriever.active["build_id"],
              "note": "Four development checks only; section hits do not prove answer completeness.",
              "checks": []}
    for name, question, expected in CASES:
        started = perf_counter()
        baseline = retriever.retrieve(question, top_k=top_k, as_of=report["as_of"])
        baseline_seconds = perf_counter() - started
        started = perf_counter()
        routed = run_routed(retriever, question, top_k=top_k, as_of=report["as_of"])
        routed_seconds = perf_counter() - started
        for mode, results, duration in [("baseline", baseline, baseline_seconds),
                                         ("routed", routed["results"], routed_seconds)]:
            for method, matches in results.items():
                evidence = score_evidence(expected, matches)
                missing = evidence["missing_sections"]
                check = {"case": name, "question": question, "mode": mode, "method": method,
                         "expected_sections": expected, **evidence, "matches": matches,
                         "seconds_for_both_namespaces": duration}
                if mode == "routed":
                    check.update({"route": routed["route"], "trace": routed["trace"]})
                report["checks"].append(check)
                print(f"{name} / {mode} / {method}: "
                      f"{'PASS' if not missing else 'MISSING ' + str(missing)}", flush=True)
    report["passed"] = sum(check["all_sections_found"] for check in report["checks"])
    report["strict_passed"] = sum(check["all_exact_sections_found"] for check in report["checks"])
    report["total"] = len(report["checks"])
    return report
