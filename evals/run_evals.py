#!/usr/bin/env python3
"""
run_evals.py — Run eval test cases and report pass/fail.

Usage:
    python -m evals.run_evals

Runs 10 test cases, checks each against expected behavior,
and prints a scorecard. A failing test we wrote ourselves tells
the evaluator more than a perfect demo.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.pipeline import RAGPipeline


def load_test_cases():
    path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    with open(path) as f:
        return json.load(f)


def check_result(test_case: dict, result: dict) -> tuple[bool, str]:
    """
    Check a result against expected behavior.
    Returns (passed, reason).

    These are heuristic checks — not perfect, but catch the most
    important failure modes.
    """
    answer = result.get("answer", "").lower()
    citations = result.get("citations", [])
    answered = result.get("answered", True)
    check_type = test_case.get("check_type", "")
    reasons = []

    # Check answered flag
    expected_answered = test_case.get("expected_answered", True)
    if answered != expected_answered:
        reasons.append(
            f"answered={answered}, expected {expected_answered}"
        )

    # Check required citations
    for cid in test_case.get("required_citations", []):
        if cid not in citations:
            # Also check if mentioned inline in answer
            if cid.lower() not in answer:
                reasons.append(f"Missing citation: {cid}")

    # Check forbidden citations
    for cid in test_case.get("forbidden_citations", []):
        if cid in citations:
            reasons.append(f"Forbidden citation present: {cid}")

    # Check forbidden phrases in answer
    for phrase in test_case.get("forbidden_in_answer", []):
        if phrase.lower() in answer:
            reasons.append(f"Forbidden phrase found: '{phrase}'")

    # Type-specific checks
    if check_type == "refusal":
        if answered:
            reasons.append("Should have set answered=false for a question the data can't answer")

    if check_type == "unit_conversion":
        # Check if conversion is mentioned
        conversion_indicators = ["semester", "per year", "annual", "convert", "×2", "x2", "× 2", "x 2", "multiply"]
        if not any(ind in answer for ind in conversion_indicators):
            reasons.append("No evidence of unit conversion in answer")

    if check_type == "must_not_cite_as_worst":
        # C006 should not be called worst
        bad_patterns = ["c006", "nainital"]
        worst_patterns = ["worst", "lowest", "poorest", "bottom"]
        if any(b in answer for b in bad_patterns) and any(w in answer for w in worst_patterns):
            # Check if it's explicitly explaining WHY it's not worst
            explain_patterns = ["not reported", "not applicable", "not meaningful", "medical"]
            if not any(e in answer for e in explain_patterns):
                reasons.append("C006 cited as worst without explaining 0 = not reported")

    if check_type == "diploma_exclusion":
        if "c005" in answer or "shivalik" in answer:
            # It's okay if mentioned with a caveat
            caveat_words = ["diploma", "not a degree", "not degree", "polytechnic", "note", "however", "caveat"]
            if not any(c in answer for c in caveat_words):
                reasons.append("C005 included without diploma caveat")

    if check_type == "about_field_cost":
        extra_cost_words = ["studio", "material", "printing", "additional", "extra", "beyond tuition",
                            "over and above", "30,000", "40,000", "thirty", "forty"]
        if not any(w in answer for w in extra_cost_words):
            reasons.append("Missing additional costs from about field (studio/material/printing charges)")

    passed = len(reasons) == 0
    return passed, "; ".join(reasons) if reasons else "OK"


def main():
    test_cases = load_test_cases()
    print(f"Running {len(test_cases)} eval test cases...\n", file=sys.stderr)

    pipeline = RAGPipeline()

    results_log = []
    passed_count = 0

    for tc in test_cases:
        print(f"  [{tc['id']}] {tc['name']}...", file=sys.stderr, end=" ")
        result = pipeline.answer(tc["query"])
        passed, reason = check_result(tc, result)

        status = "✅ PASS" if passed else "❌ FAIL"
        print(status, file=sys.stderr)

        results_log.append({
            "id": tc["id"],
            "name": tc["name"],
            "query": tc["query"],
            "passed": passed,
            "reason": reason,
            "system_answer": result.get("answer", "")[:200],
            "system_citations": result.get("citations", []),
            "system_answered": result.get("answered"),
        })

        if passed:
            passed_count += 1

    # Print scorecard
    total = len(test_cases)
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"EVAL SCORECARD: {passed_count}/{total} passed ({100 * passed_count / total:.0f}%)", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

    # Print failures
    failures = [r for r in results_log if not r["passed"]]
    if failures:
        print("FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  [{f['id']}] {f['name']}: {f['reason']}", file=sys.stderr)
            print(f"    Answer preview: {f['system_answer'][:150]}...", file=sys.stderr)
            print(file=sys.stderr)

    # Write results to file
    with open("evals/eval_results.json", "w") as fout:
        json.dump({
            "total": total,
            "passed": passed_count,
            "pass_rate": f"{100 * passed_count / total:.0f}%",
            "results": results_log,
        }, fout, indent=2, ensure_ascii=False)

    print(f"Detailed results written to evals/eval_results.json", file=sys.stderr)

    # Print summary to stdout
    print(json.dumps({
        "total": total,
        "passed": passed_count,
        "pass_rate": f"{100 * passed_count / total:.0f}%",
        "failures": [{"id": f["id"], "name": f["name"], "reason": f["reason"]} for f in failures],
    }, indent=2))


if __name__ == "__main__":
    main()
