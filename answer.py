#!/usr/bin/env python3
"""
answer.py — CLI entry point for the MME RAG prototype.

Usage:
    python answer.py "Which colleges offer an MBA, and what do they cost?"

Outputs a single JSON object to stdout:
    {
        "answer": "...",
        "citations": ["C002", "C004"],
        "answered": true,
        "reason_if_unanswered": null
    }
"""

import sys
import json
from src.pipeline import RAGPipeline


# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python answer.py \"<your question>\"", file=sys.stderr)
#         sys.exit(1)

#     query = sys.argv[1]

#     # Check for --measure-cost flag
#     measure_cost = "--measure-cost" in sys.argv

#     pipeline = RAGPipeline()
#     result = pipeline.answer(query, measure_cost=measure_cost)

#     # Print clean JSON to stdout
#     print(json.dumps(result, indent=2, ensure_ascii=False))


def interactive():
    """REPL mode — ask questions one at a time."""
    print("MME College Counsellor — interactive mode")
    print("Type your question, or 'quit' to exit.\n")

    pipeline = RAGPipeline()
    print("Ready.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        result = pipeline.answer(query, measure_cost=True)
        metrics = result.pop("_metrics", {})

        print(f"\nBot: {result['answer']}")
        if result.get("citations"):
            print(f"Sources: {', '.join(result['citations'])}")
        if not result.get("answered", True):
            print(f"(unanswered: {result.get('reason_if_unanswered')})")
        if metrics:
            print(f"[{metrics.get('input_tokens')} in / {metrics.get('output_tokens')} out "
                  f"| {metrics.get('latency_seconds')}s | ${metrics.get('cost_usd')}]")
        print()


def main():
    # Interactive mode
    if len(sys.argv) > 1 and sys.argv[1] in ("-i", "--interactive"):
        interactive()
        return

    if len(sys.argv) < 2:
        print("Usage: python answer.py \"<your question>\"", file=sys.stderr)
        print("       python answer.py --interactive", file=sys.stderr)
        sys.exit(1)

    query = sys.argv[1]
    measure_cost = "--measure-cost" in sys.argv

    pipeline = RAGPipeline()
    result = pipeline.answer(query, measure_cost=measure_cost)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
