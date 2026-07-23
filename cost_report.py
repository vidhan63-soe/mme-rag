#!/usr/bin/env python3
"""
cost_report.py — Measure and report cost metrics (Part D).

Runs a representative set of queries and reports:
- Average input/output tokens per query
- Average end-to-end latency
- Cost per 1,000 queries in ₹
- One-time embedding cost
"""

import json
import sys
import time
from src.pipeline import RAGPipeline
from src.config import get_provider, MODELS

SAMPLE_QUERIES = [
    "I scored 78% and have a budget of ₹1.5 lakh/year — which engineering colleges can I consider?",
    "Which colleges offer an MBA, and what do they cost?",
    "List the government colleges that have hostel facilities.",
    "What's the average placement package at North Ridge Institute of Technology?",
    "Does Ganga Valley University offer a PhD in Physics?",
    "Which colleges offer scholarships for students from low-income families?",
    "Which college is best for me? I have ₹1 lakh per semester.",
]

USD_TO_INR = 85.0  # approximate


def main():
    provider, _ = get_provider()
    model_cfg = MODELS[provider]

    print(f"Provider: {provider}", file=sys.stderr)
    print(f"Model: {model_cfg['model']}", file=sys.stderr)
    print(f"Running {len(SAMPLE_QUERIES)} queries for cost measurement...\n", file=sys.stderr)

    pipeline = RAGPipeline()

    # Measure embedding time
    t_embed_start = time.perf_counter()
    # Re-init to measure fresh embedding
    from src.retriever import HybridRetriever
    _ = HybridRetriever()
    t_embed = time.perf_counter() - t_embed_start

    metrics_list = []
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        print(f"  [{i}/{len(SAMPLE_QUERIES)}] Running...", file=sys.stderr)
        result = pipeline.answer(q, measure_cost=True)
        if "_metrics" in result:
            metrics_list.append(result["_metrics"])

    if not metrics_list:
        print("ERROR: No metrics collected. Check API keys.", file=sys.stderr)
        sys.exit(1)

    # Aggregate
    n = len(metrics_list)
    avg_input = sum(m["input_tokens"] for m in metrics_list) // n
    avg_output = sum(m["output_tokens"] for m in metrics_list) // n
    avg_latency = sum(m["latency_seconds"] for m in metrics_list) / n
    avg_cost_usd = sum(m["cost_usd"] for m in metrics_list) / n

    cost_per_1k_usd = avg_cost_usd * 1000
    cost_per_1k_inr = cost_per_1k_usd * USD_TO_INR

    report = {
        "provider": provider,
        "model": model_cfg["model"],
        "cost_per_1m_input_usd": model_cfg["cost_per_1m_input"],
        "cost_per_1m_output_usd": model_cfg["cost_per_1m_output"],
        "sample_size": n,
        "avg_input_tokens_per_query": avg_input,
        "avg_output_tokens_per_query": avg_output,
        "avg_latency_seconds": round(avg_latency, 3),
        "cost_per_1000_queries_usd": round(cost_per_1k_usd, 4),
        "cost_per_1000_queries_inr": round(cost_per_1k_inr, 2),
        "one_time_embedding_cost_usd": 0.0,
        "one_time_embedding_time_seconds": round(t_embed, 2),
        "embedding_model": "all-MiniLM-L6-v2 (local, free)",
        "notes": (
            "Embedding uses a local sentence-transformers model — zero API cost. "
            f"Embedding 15 colleges took {t_embed:.2f}s. "
            "At 50K queries/month the LLM API cost is the dominant factor."
        ),
    }

    # Print table
    print("\n" + "=" * 60)
    print("COST REPORT — Part D")
    print("=" * 60)
    print(f"{'Metric':<40} {'Value':>18}")
    print("-" * 60)
    print(f"{'Model':<40} {report['model']:>18}")
    print(f"{'Avg input tokens/query':<40} {report['avg_input_tokens_per_query']:>18}")
    print(f"{'Avg output tokens/query':<40} {report['avg_output_tokens_per_query']:>18}")
    print(f"{'Avg latency/query':<40} {report['avg_latency_seconds']:>16.3f}s")
    print(f"{'Cost per 1M input tokens (USD)':<40} ${report['cost_per_1m_input_usd']:>16.2f}")
    print(f"{'Cost per 1M output tokens (USD)':<40} ${report['cost_per_1m_output_usd']:>16.2f}")
    print(f"{'Cost per 1,000 queries (USD)':<40} ${report['cost_per_1000_queries_usd']:>16.4f}")
    print(f"{'Cost per 1,000 queries (INR)':<40} ₹{report['cost_per_1000_queries_inr']:>16.2f}")
    print(f"{'One-time embedding cost':<40} {'₹0 (local model)':>18}")
    print(f"{'Embedding time (15 colleges)':<40} {report['one_time_embedding_time_seconds']:>16.2f}s")
    print("=" * 60)

    # Write JSON
    with open("cost_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nDetailed report saved to cost_report.json")


if __name__ == "__main__":
    main()
