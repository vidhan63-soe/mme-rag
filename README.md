# MME RAG Prototype - Make My Education AI Counsellor

A grounded, citation-backed RAG system that answers student questions about colleges using only verified data. Built for the Applied AI Engineer take-home assignment.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your LLM API key (copy .env.example to .env and fill it in)
cp .env.example .env
# then edit .env:  GROQ_API_KEY=gsk_...
# Get a free key at console.groq.com. OPENAI_API_KEY also works.

# 3. Ask a question
python answer.py "Which colleges offer an MBA, and what do they cost?"

# 4. Interactive mode (pipeline stays warm between questions)
python answer.py --interactive

# 5. Regenerate answers.md for the 7 published questions
python run_all.py

# 6. Run the evaluation suite
python -m evals.run_evals

# 7. Cost measurement (Part D)
python cost_report.py
```

---

## The Core Design Decision

Early in development the system produced two answers that were confidently wrong:

> "your budget of ₹1.5 lakh/year is below the annual tuition fees of both colleges: ₹132,000 for C003 and ₹118,000 for C009. Neither college fits within your budget."

> "By average placement: C012 (₹8.4 LPA) < C004 (₹7.1 LPA)"

Retrieval was correct in both cases. The right colleges were fetched with the right numbers attached. The LLM then compared those numbers in prose and got the comparison backwards.

In admissions, a wrong fee destroys trust. Prompting harder is not a fix - it lowers the failure rate without removing the failure mode. **So the system no longer asks the LLM to do arithmetic.**

Every numeric comparison and every ordering is computed in Python and injected into the prompt as a finished verdict:

```
⚠️ PRE-COMPUTED FILTER RESULTS (trust these, do NOT re-compare):
  C003: Fee ₹132,000 vs budget ₹150,000 → ✅ WITHIN BUDGET | Cutoff 75% vs score 78% → ✅ ELIGIBLE
  C009: Fee ₹118,000 vs budget ₹150,000 → ✅ WITHIN BUDGET | Cutoff 70% vs score 78% → ✅ ELIGIBLE

📊 PRE-COMPUTED RANKINGS (already sorted - reproduce as-is, do NOT re-order):
  By annual fee, lowest to highest: C007 (₹15,000) < C012 (₹45,000) < C014 (₹72,000)
  By average placement, highest to lowest: C012 (₹8.4 LPA) > C004 (₹7.1 LPA)
  Placement not reported (excluded from ranking, NOT 'worst'): C006
```

The LLM writes prose around conclusions it did not calculate. Both failures disappeared and have not recurred.

This also handles the `avg_placement_lpa = 0` trap structurally rather than by instruction: colleges with unreported placement are removed from the ranking in Python and labelled explicitly, so the model cannot rank C006 last even if it wanted to.

---

## Architecture

```
query
  ↓
query_analyzer.py    classify: listing / lookup / semantic
                     flag: unit ambiguity, subjectivity, needs about-field
  ↓
retriever.py         Stage 1 - structured filters (exact, on DataFrame)
                     Stage 2 - semantic similarity (local embeddings)
                     Stage 3 - combine: filters decide who, similarity decides order
  ↓
pipeline.py          pre-compute verdicts and rankings in Python
                     inject as authoritative notes
  ↓
generator.py         LLM call, strict JSON out, citations required
```

### Hybrid retrieval, and why not pure vector search

**Structured filtering** extracts hard constraints via rules - budget, cutoff score, course, college type, hostel, named college - and applies them as exact masks on the DataFrame.

Cosine similarity has no concept of `≤`. A ₹8.5 lakh college embeds close to "budget ₹1.5 lakh" because both contain "lakh". Filtering is exact; similarity is not.

**Semantic search** embeds the query and ranks against 15 pre-embedded college documents. Each document merges every structured field into readable prose alongside the full `about` text, so semantic search can surface a college by any attribute.

Questions like *"which colleges offer scholarships for low-income families?"* have zero structured fields to filter on. The answer lives entirely in free text.

**Combination.** If filters narrowed the set, that set is used and similarity only orders it. If no filter fired, top-K by similarity. A college named explicitly in the query is always included even when other filters exclude it - *"Does Ganga Valley University offer a PhD?"* needs C002's record present for the model to answer "no."

### Negation handling

`"colleges with a hostel"` and `"colleges without a hostel"` both contain the token `hostel`. Treating them identically returns the exact opposite set - and the model, seeing only colleges that *do* have hostels, confidently reports that none lack one. This was caught by eval_08 and fixed by detecting negation words near the token.

The tri-state matters: `hostel_required` can be `True`, `False`, or absent, and `.get()` collapses `False` into absent. Membership is tested with `in`, not `.get()`.

**Known limitation:** negation is handled for hostel only. `"colleges not in Dehradun"` or `"not government"` will be filtered as though the negation were absent. Documented rather than silently broken.

### Model choices

| Component | Choice | Reasoning |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2`, local | 15 docs embed in under 2s. Zero API cost. No provider dependency for retrieval. |
| LLM | Llama 3.3 70B via Groq | Free tier, sub-second typical latency, reliable JSON mode. |
| Fallback | OpenAI `gpt-4o-mini` | Auto-selected if `OPENAI_API_KEY` is set instead. |

I also ran the full suite against Llama 3.1 8B on the same Groq endpoint. It is roughly an order of magnitude cheaper and passes most cases, but it fails on precisely the kind of instruction this system depends on - it answered the published PhD question correctly in prose while setting `answered: true`, which is the flag the brief specifically asks to be right. The 70B model gets that case right. That is a concrete example of what the extra cost buys.

### Vector store: deliberately not used

At 15 records, semantic search is one matrix multiply over a `(15, 384)` array - microseconds. ChromaDB or FAISS would add a dependency, a persistence layer, and an initialisation step to return the identical top-K. An ANN index solves a scan cost that does not exist at this scale.

I have run ChromaDB in production on a knowledge base several orders of magnitude larger (see Part B), which is part of why I am comfortable saying it is the wrong tool here. Past roughly 1,000 colleges the calculation reverses: sub-linear search starts to pay, and persistence removes the cold-start re-embedding cost. The swap is contained - `_extract_filters`, the verdict injection, and the prompt are all unaffected by it.

### Techniques considered and rejected

**Reciprocal Rank Fusion.** RRF merges ranked lists from multiple retrievers, typically BM25 and dense embeddings. There is one ranker here; fusing a list with itself is the identity operation. Adding BM25 to make RRF meaningful would fuse two rankers that, on 15 documents already narrowed to 2–5 candidates by structured filtering, agree on essentially everything. Worth adding once a second retriever earns its place - call it 500+ records.

**Query expansion.** Rewriting a query into paraphrases costs an extra LLM call, roughly doubling latency and per-query cost, on a system where the structured filter already extracted the constraint exactly. It also introduces terms the student did not say - and on a dataset containing both *Ganga Valley University* (C002) and *Ganga Institute of Commerce* (C014), a paraphrase that drops the distinguishing word is precisely how those two get confused. Expansion earns its cost when students use vocabulary the data lacks; that is a content problem, not a scale one.

**Mean Reciprocal Rank** is a metric rather than a technique, and it is worth computing - the eval cases already declare `required_citations`, so MRR over them is a small addition. Not yet implemented; noted as a next step.

---

## Evaluation

`python -m evals.run_evals` → **9/10 passing (90%)**

Ten cases, each targeting a specific trap in the data dictionary rather than a happy path.

| ID | Tests | Result |
|---|---|---|
| eval_01 | `placement = 0` is "not reported", not "worst" | ✅ |
| eval_02 | Refusal - course absent from dataset | ❌ *(see below)* |
| eval_03 | Semester → annual fee conversion | ✅ |
| eval_04 | Diploma is not a degree (C005 exclusion) | ✅ |
| eval_05 | Similar-name disambiguation (C002 vs C014) | ✅ |
| eval_06 | Cutoff as a hard floor, no hedging | ✅ |
| eval_07 | Costs beyond tuition, from the `about` field | ✅ |
| eval_08 | Negated constraint - government *without* hostel | ✅ *(was failing)* |
| eval_09 | Field absent from schema | ✅ *(test corrected - see below)* |
| eval_10 | Total-course-cost → per-year conversion | ✅ |

**eval_08 was a genuine retrieval bug.** The query *"government colleges without hostel facilities"* returned C007 and C012 - the colleges that *do* have hostels - and the system reported that none lacked one. False, and confidently so. Fixed by negation detection; now correctly returns C005 and C011.

**eval_02 is a real system failure and I have left it in.** Asked *"which colleges offer a B.Sc in Agriculture?"*, the system correctly answers that none do, and helpfully notes that C007 offers B.Sc in other science streams - then sets `answered: true`. The prose is right; the flag is wrong. The model appears to be answering "did I produce useful text?" rather than "was the student's question answered?".

I tried two fixes. An explicit prompt rule (*"mentioning a nearby option does not make the question answered"*) reduced but did not eliminate it. A deterministic Python override was then added, scoped to relaxed retrieval with a course filter - but this query produces no course filter at all, because "Agriculture" is not a term the extractor knows, so retrieval never relaxes and the override never fires. Fixing it properly means either enumerating courses that are *not* in the data, which is unbounded, or validating the flag against the answer text. The latter is the right approach and is item 7 below.

**eval_09 is a case where my test was wrong and the system was right.** I asked for a student-to-faculty ratio, which does not exist in the schema, and wrote the expectation as `answered: true` on the assumption that a partial answer from the `about` field would be acceptable. The system refused instead and set `answered: false`. That is the better judgment: a ratio is a specific numeric claim, and the `about` field offers only a rough faculty count for a different college. Answering partially would invite the student to infer a number that was never in the data. The test expectation has been corrected to match, and the reasoning is recorded here rather than quietly edited away.

The suite sits at 9/10 and I have not tuned it to pass. eval_02 is a real defect with a documented cause; a suite where everything passes is a suite that is too easy.

---

## Part B - Proof You've Shipped

### Multi-Agent RAG for R&D Knowledge Retrieval - BioGenex

**What it did, and who used it.** An internal RAG assistant for BioGenex's R&D department, deployed on GCP. Researchers queried it against company documentation - antibody and probe specifications, protocols, and a wider enterprise knowledge base - to pull precise technical details during experiments instead of searching PDFs by hand. The users were the R&D team internally; this was never a customer-facing product.

**On evidence.** The system was built inside the company, and both the source and the underlying documents are confidential, so I cannot share a repository or a link. What I can do is describe the architecture and the production failures in as much detail as you want, and answer questions on either.

**What I personally built.** I wrote the retrieval pipeline end to end: Sentence-Transformers for embeddings, ChromaDB as the vector store, and LangChain agent orchestration over GPT-4 for context-aware retrieval. I implemented the multi-agent workflow - separate retrieval, web search, and verification agents using ReAct and function calling - along with the source-grounding and confidence-scoring layer on top. I also built the service around it: FastAPI with async processing, Redis caching for repeated queries, and PostgreSQL for conversation history. On a related project for the same team I integrated a voice layer using Edge TTS and local models via Ollama, giving verbal status updates during instrument runs.

**What broke in production.** Two things, and the second one caught me out.

The first was retrieval quality. Measured against what the R&D team judged a query *should* have returned, MRR was well below what the demo suggested. Chunks that looked plausible in isolation were the wrong chunks for the question actually asked. This is a familiar failure once you have seen it, but it does not surface until domain experts bring real questions - my own spot checks had all passed.

The second was cold-start latency. In chat and research use a single question could trigger several agent hops, each with its own retrieval and LLM call. Steady-state performance was fine, but the first request after a cold start took long enough to exceed the request timeout and fail outright. Warm caches and an already-running service had hidden this throughout testing, which is exactly why it only appeared in production. The fix worked on two fronts: cutting the number of sequential agent hops on the common path, and caching hard enough that the expensive path ran rarely rather than routinely.

That experience is the direct reason this prototype does as much work as it can outside the model. Every extra LLM call is another opportunity to be slow and another opportunity to be wrong.

**What it cost to run.** Little, in direct terms. Embeddings ran on local Sentence-Transformers models and parts of the stack used open-source models through Ollama, so recurring spend was essentially GCP hosting rather than per-token API cost - the same reasoning behind the local embedding model here. The genuinely expensive component was GPT-4 inside the multi-agent loop, so reducing hops per query was as much a cost decision as a latency one.

### Related work

- **AI Pre-Screening Recruitment Agent with Live Voice Interview** - a full-stack product pairing a ChromaDB RAG pipeline for resume-to-JD ranking with a WebRTC voice interview agent (Whisper for STT, ElevenLabs/Bark for TTS), plus a hybrid scoring engine blending cosine retrieval with LLM evaluation. Code available on request.
- **LincolnAI (Aug 2024 – Jun 2025)** - a RAG agent for contract drafting and legal document review built on LangChain, GPT-4, and FAISS over a policy-document knowledge base, served through FastAPI.

---

## Part C - Written Reflection

**Keeping per-query cost low as usage grows.**
Three levers, in order of impact. First, semantic caching - hash the query embedding and serve a cached answer when a sufficiently similar query was handled recently. College data changes slowly and student questions cluster hard around a few dozen patterns, so caching should absorb a large share of traffic before it reaches the model; this was also the lever that mattered most at BioGenex. Second, tiered routing: the query analyzer already classifies complexity, so simple lookups ("fees at X") can go to a small model while genuinely comparative questions get a larger one - though see the model-choice note above on what the cheaper model actually costs you in accuracy. Third, keep it to one LLM call per query wherever possible - multi-agent loops multiply cost and latency together, and most student questions do not need them.

**Never stating a wrong fee or cutoff.**
The system must never *generate* a number, only copy one. This prototype takes the first step by removing arithmetic from the LLM's job: comparisons and rankings are computed in Python and injected as verdicts. The next step is a post-generation validator that extracts every ₹ amount and percentage from the response and blocks it unless each appears in the retrieved records. Structured fields should be injected as structured data rather than embedded prose, so there is no parsing layer between the database and the claim. Unit labels ("per year") should be enforced by the validator, not left to the model's discretion.

**What I would build first.**
The unit and ambiguity layer, before touching retrieval. Students speak in per-semester fees, total course cost, lakhs, and percentages interchangeably, and silently answering a per-semester question with a per-year figure is a trust failure that no amount of retrieval quality compensates for. A normaliser that detects ambiguity, asks when it is genuinely unresolvable, and shows its conversion when it resolves it, is the foundation everything else sits on.

**Measuring whether AI is actually helping.**
Shortlist-or-apply completion rate for students who use the counsellor against those who do not. A weekly human audit of sampled responses scored for factual correctness, tracked as an error rate over time - at BioGenex the gap between demo-quality retrieval and what domain experts judged correct was large, and only human-scored sampling surfaced it. Refusal rate as a health signal: too high means the data is thin, too low means the system is guessing, and somewhere in the 5–15% band is honest. Escalation rate to human counsellors, which should fall as the system improves. And one question to the student after the session: did this help you decide?

---

## Part D - Cost, With Numbers

Measured across the 7 published questions via `run_all.py`. Reproduce with `python cost_report.py`.

| Metric | Value |
|---|---|
| Average input tokens per query | 1,842 |
| Average output tokens per query | 142 |
| Average end-to-end latency per query | 6.72 s |
| Model | Llama 3.3 70B (Groq) |
| Cost per 1M input tokens | $0.59 |
| Cost per 1M output tokens | $0.79 |
| Total cost, 7 queries | $0.0084 |
| **Cost per 1,000 queries** | **~$1.20 (≈ ₹102)** |
| One-time embedding cost | ₹0 - local model, ~2 s for 15 colleges |

The 6.72 s average is skewed by two outliers (15.5 s and 20.7 s) on the full-context questions; the median is closer to 0.6 s. Input tokens scale with how many colleges survive filtering - a targeted lookup runs around 1,000 tokens, a full-scan scholarship question around 4,288.

### At 50,000 queries/month, what breaks first?

**Not cost.** ₹102 per 1,000 queries puts 50K/month at roughly ₹5,100 - negligible against the value of a counselling interaction.

**Latency is the first real constraint,** and it bites before cost does. Groq's free tier capped this prototype at 100K tokens/day, which I hit during development. At ~1,840 input tokens per query, 50K queries is roughly 92M tokens a month - well past free-tier limits and into rate-limiting during peak hours, which in admissions means the weeks around results. First fix is semantic caching, which should cut model calls substantially given how tightly student questions cluster, plus routing simple lookups to a smaller model. Both reduce latency and cost at once.

**Accuracy is the risk that actually matters.** At 50K queries students will find every edge case - negations the filter misses, courses absent from the data, Hindi and Hinglish phrasing, questions about hostel food and campus life that no structured field answers. The eval suite already caught one confident-wrong answer (eval_08), one open defect in refusal labelling (eval_02), and one place where my own test was worse than the system's judgment (eval_09). Before optimising anything else I would expand that suite, add the numeric post-generation validator, and instrument the refusal rate in production as a live health signal.

---

## What I'd Do Differently With More Time

1. **Post-generation numeric validator.** Extract every ₹ figure and percentage from the answer, verify each against the retrieved records, block on mismatch. This would have caught the original arithmetic failures automatically rather than by inspection.

2. **Multi-turn conversation.** Currently single-shot by design, matching the required interface. Real counselling is a conversation - "what about engineering?" then "which of those have hostels?" then "cheapest one?". The hard part is not history in the prompt but history-aware *query rewriting* before retrieval: "cheapest one?" has to become "cheapest among C003 and C009" or the retriever fetches the wrong colleges entirely.

3. **General negation handling.** Currently hostel-only; should extend to location, type, and course constraints.

4. **Hindi and Hinglish.** Most students in Uttarakhand would rather ask in Hindi. Detect, translate, retrieve, translate back.

5. **SQLite for structured fields.** Replaces the pandas filter path with indexed SQL - faster, auditable, and the natural home for the structured half of the hybrid as the dataset grows.

6. **MRR on the eval set,** to put a retrieval number alongside the generation pass rate.

7. **Validate the `answered` flag against the answer text.** eval_02 shows the model can write a correct refusal and then mislabel it. A check that flags disagreement between refusal-shaped prose and `answered: true` would catch this class of bug generally, rather than patching each individual trigger.

---

## Repository Structure

```
├── answer.py              # CLI - single-shot JSON, plus --interactive REPL
├── run_all.py             # Regenerates answers.md
├── cost_report.py         # Part D measurement
├── requirements.txt
├── .env.example           # Template - copy to .env and add your key
├── README.md
├── answers.md             # Verbatim output, 7 published questions
├── sample_colleges.csv
├── DATA_DICTIONARY.md
├── src/
│   ├── config.py          # Provider routing, model + pricing config
│   ├── data_loader.py     # CSV → rich text documents
│   ├── retriever.py       # Hybrid structured + semantic retrieval
│   ├── query_analyzer.py  # Classification and routing flags
│   ├── generator.py       # Grounded generation, strict JSON
│   └── pipeline.py        # Orchestration + pre-computed verdicts
└── evals/
    ├── test_cases.json    # 10 cases targeting data-dictionary traps
    ├── run_evals.py       # Runner + scorecard
    └── eval_results.json  # Last run: 9/10
```
