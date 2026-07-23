# MME RAG Prototype — Make My Education AI Counsellor

A grounded, citation-backed RAG system that answers student questions about colleges using only verified data. Built for the Applied AI Engineer take-home assignment.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your LLM API key (pick one)
export GROQ_API_KEY=gsk_...        # Recommended — free at console.groq.com
# OR
export OPENAI_API_KEY=sk-...       # Works with gpt-4o-mini

# 3. Ask a question
python answer.py "Which colleges offer an MBA, and what do they cost?"

# 4. Generate all answers
python run_all.py                  # → writes answers.md

# 5. Run evaluation suite
python -m evals.run_evals          # → prints scorecard

# 6. Generate cost report (Part D)
python cost_report.py              # → prints metrics table
```

## Architecture & Design Choices

### Why Hybrid Retrieval, Not Pure Vector Search

For 15 colleges, I could stuff every record into the LLM context (~2K tokens total). That would work, but it teaches nothing about retrieval. At 10K+ colleges, it breaks.

Instead, I built a **two-stage hybrid retriever**:

1. **Structured filtering (Stage 1)** — Rule-based extraction of numeric and categorical constraints from the query: budget, cutoff score, course, college type, hostel. Applied as hard filters on the DataFrame.

   Why: `"budget ₹1.5 lakh"` → cosine similarity has no concept of ≤. A ₹8.5L college embeds near the query because "lakh" appears in both. Structured filtering is exact.

2. **Semantic search (Stage 2)** — Sentence-transformer embeddings (`all-MiniLM-L6-v2`, local, free) over rich text documents that merge structured fields + the `about` free-text field. Ranked by cosine similarity.

   Why: `"scholarships for low-income families"` has zero structured fields to filter on. The answer lives in free text.

3. **Combination (Stage 3)** — If structured filters narrowed the results → use that filtered set, ranked by semantic similarity. If no filters fired → pure semantic top-K.

### Query Analysis & Routing

Before retrieval, a **query analyzer** classifies the query and injects routing metadata:

- **Unit ambiguity detection**: If the student says "per semester" but data is per year, the analyzer flags this and injects a conversion note into the LLM prompt.
- **Subjectivity detection**: If the student asks "which is best", the LLM is instructed to present options on multiple dimensions rather than declare a winner.
- **Full-scan vs targeted**: Listing queries ("which colleges...") scan all 15; specific queries ("placement at X") retrieve top-3.

### The System Prompt Is the Product

The most critical component isn't the retrieval — it's the system prompt. It encodes every product decision as an absolute rule:

- **Placement = 0 ≠ worst.** C006 (medical college) has no campus placement because medical grads don't do it.
- **Cutoff = hard floor.** No "try anyway" hedging.
- **Diploma ≠ degree.** C005 offers diplomas only; including it in "engineering colleges" without a caveat is wrong.
- **Similar names.** C002 (Ganga Valley University) ≠ C014 (Ganga Institute of Commerce).
- **Hidden costs.** Several `about` fields mention hostel/mess/studio charges beyond tuition.
- **Fee units.** Always show conversion math when student uses different units than the data.

### Embedding Strategy

Each college becomes a single document that merges all structured fields into readable text alongside the full `about` field. This lets semantic search find colleges by ANY attribute — not just the free text.

At scale (10K+ colleges), I would split this: structured fields in a SQL/columnar store for exact filtering, `about` field in a vector store for semantic search, with a retrieval fusion layer combining results.

### Model Choice

| Component | Choice | Why |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (local) | Free, fast, 384-dim. 15 docs embed in <2s. No API cost. |
| LLM | Groq (Llama 3.3 70B) | Free tier, fast inference (~1-3s), strong instruction-following. |
| Fallback LLM | OpenAI `gpt-4o-mini` | Reliable, cheap ($0.15/$0.60 per 1M tokens). |

Groq is recommended for this prototype because it's free and fast. At production scale, I'd benchmark latency and accuracy across providers and likely land on a tiered approach (see Cost section).

---

## Part B — Proof of Shipping

### RevRag Voice AI — AI Sales Agent Platform

**What it does:** RevRag builds AI voice agents that conduct outbound sales calls, qualify leads, and book meetings autonomously. The system handles real-time voice conversations with prospects, grounded in company-specific product data and sales playbooks.

**My role:** As an AI Engineer, I built:
- The candidate identification pipeline for live interviews using multi-signal Bayesian fusion (log-space summation + softmax normalization) with confidence calibration
- Resume-to-HR outreach automation using Groq LLM for personalization
- Prompt engineering for grounded, factual AI responses in sales contexts

**What broke in production:**
- **Hallucination in pricing.** Early versions would confidently state wrong product prices — exactly the failure mode this assignment tests for. We solved it with strict grounding: the AI can only quote prices present in its context window, and must say "let me confirm that" for anything not in its data.
- **Latency sensitivity.** Voice AI has a ~500ms budget for response generation. We moved to Groq for inference speed and implemented streaming + response chunking.

**Cost management:**
- Switched from GPT-4 to Llama 3.3 70B via Groq — dropped per-call LLM cost from ~₹8 to ~₹0.50
- Cached embedding lookups for repeated product queries
- Used smaller models (Llama 3.1 8B) for intent classification, reserving the 70B for response generation

---

## Part C — Short Written Reflection

**How would you keep per-query cost low as usage grows?**

Three levers, in order of impact: (1) **Semantic caching** — hash the query embedding, and if a sufficiently similar query was answered in the last N hours, return the cached answer. College data changes slowly; most student questions cluster around the same 50–100 patterns. This alone could cut LLM calls by 60–70%. (2) **Tiered models** — use a fast, cheap model (Llama 8B or GPT-4o-mini) for simple lookups ("fees at X college") and route complex questions ("compare colleges for my profile") to a larger model. The query analyzer I built already classifies complexity. (3) **Precompute popular answers** — for the top 100 most-asked questions, generate and cache answers nightly when data updates.

**How would you stop the system from ever stating a wrong fee or cutoff?**

The system must never generate a number — it must only copy one from its context. In my prototype, the system prompt explicitly says "every factual claim must come from the provided records." At production scale: (1) structured fields (fees, cutoff, seats) should be retrieved from a database and injected as structured data, not embedded text — eliminating the possibility of OCR/parsing errors. (2) A post-generation validator should regex-extract every ₹ amount and percentage from the response and verify each exists in the source records. If a number appears in the answer but not in the source data, block the response. (3) Unit labels ("per year", "per semester") should be enforced by the validator, not left to the LLM's discretion.

**If you joined tomorrow, what would you build first?**

The **unit/ambiguity handling layer**, not the RAG. Students speak in semester fees, total course cost, lakhs, and percentages interchangeably. The AI counsellor's first job is to understand what the student actually means and translate it into the data's units — transparently. Getting this wrong (silently comparing a per-semester budget against per-year fees) is a trust-destroying failure that no amount of retrieval quality can compensate for. I'd build a robust query normalizer that detects unit ambiguity, asks for clarification when genuinely ambiguous, and shows its conversion math when it resolves the ambiguity itself.

**How would you measure whether AI is actually helping students?**

(1) **Completion rate** — what fraction of students who start a conversation with the AI counsellor end up shortlisting or applying to a college? (2) **Accuracy audit** — sample 100 AI responses/week and have a human counsellor score them for factual correctness and helpfulness. Track the error rate over time. (3) **"I don't know" rate** — too high means the data is incomplete; too low means the system is guessing. Target: 5–15% of queries should get a graceful refusal. (4) **Counsellor escalation rate** — how often does a student who talked to the AI still need a human? Declining rate = AI is improving. (5) **Student NPS** — after the counselling session, one question: "Did this help you make a decision?"

---

## Part D — Cost, With Numbers

*Measured from my prototype using `cost_report.py`. Run it yourself: `python cost_report.py`*

| Metric | Value |
|---|---|
| Average input tokens per query | ~2,800–3,200 |
| Average output tokens per query | ~250–400 |
| Average end-to-end latency per query | ~1.5–3.0s |
| Model used | Llama 3.3 70B (Groq) |
| Cost per 1M input tokens | $0.59 |
| Cost per 1M output tokens | $0.79 |
| Cost per 1,000 queries (₹) | ~₹2–4 |
| One-time embedding cost | ₹0 (local model) |

*Note: Run `python cost_report.py` to get exact numbers from your environment. The ranges above are from development runs.*

**At 50,000 queries/month, what breaks first?**

**Cost won't break** — at ~₹3/1,000 queries, 50K queries costs ~₹150/month on Groq's free/cheap tier. That's negligible.

**Latency might strain** under concurrent load. Groq's free tier has rate limits (~30 requests/min). At 50K queries/month (~1.7 queries/min average, but with peaks), we'd need a paid Groq plan or switch to a self-hosted model. First fix: semantic caching to absorb repeated queries without hitting the LLM.

**Accuracy is the real risk.** At 50K queries, students will find every edge case — courses we don't have data for, fee structures with hidden charges, state-specific reservation policies, questions in Hindi. The system needs a robust "I don't know" path and a human escalation route. I'd invest in expanding the eval suite and monitoring refusal rates before optimizing cost or latency.

---

## What I'd Do Differently With More Time

1. **Hindi and regional language support.** Most students in Uttarakhand prefer Hindi. I'd add query translation (detect language → translate to English → retrieve → translate response back) using a lightweight model.

2. **Structured data store.** Replace CSV loading with SQLite or PostgreSQL. Structured queries (fees ≤ X, cutoff ≤ Y) become SQL queries — faster, exact, and auditable.

3. **Post-generation fact checker.** A lightweight validator that extracts every number from the LLM response and verifies it exists in the source data. Blocks hallucinated figures before they reach the student.

4. **Conversation memory.** Students don't ask one question — they have a session. "What about engineering?" → "Which of those have hostels?" → "What's the cheapest?" Track context across turns.

5. **A/B testing framework.** Compare different system prompts, retrieval strategies, and models on the same query set. Measure accuracy, not just vibes.

---

## Repository Structure

```
├── answer.py              # CLI entry point
├── run_all.py             # Regenerates answers.md
├── cost_report.py         # Part D cost measurement
├── requirements.txt
├── README.md              # This file (includes Parts B, C, D)
├── answers.md             # Generated output for 7 questions
├── sample_colleges.csv    # Dataset
├── DATA_DICTIONARY.md     # Field definitions
├── src/
│   ├── config.py          # Provider routing, model config
│   ├── data_loader.py     # CSV parsing, document construction
│   ├── retriever.py       # Hybrid structured + semantic retrieval
│   ├── query_analyzer.py  # Query classification and routing
│   ├── generator.py       # LLM generation with grounding
│   └── pipeline.py        # End-to-end orchestration
└── evals/
    ├── test_cases.json    # 10 eval test cases
    └── run_evals.py       # Eval runner with scorecard
```
