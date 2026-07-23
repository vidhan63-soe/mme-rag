"""
Main RAG pipeline: query analysis → retrieval → generation.

Orchestrates the three stages and handles edge cases like
unit ambiguity, subjective queries, and over-constrained retrieval
by injecting guidance into the LLM context.

Design note on pre-computation
------------------------------
An LLM asked to compare or sort numbers inside prose gets it wrong often
enough to matter — we observed "₹8.4 LPA < ₹7.1 LPA" and "₹1,32,000 is
above ₹1,50,000" in early runs. In admissions, a wrong fee destroys trust.
So every numeric comparison and ordering is computed in Python and handed
to the LLM as a finished verdict. The LLM writes prose; it never does math.

The same principle extends past arithmetic to the `answered` flag. Asked
about a course no college offers, the model writes a correct refusal and
then sets answered=true anyway, because it produced text it considers
useful. But the flag describes whether the STUDENT'S QUESTION was answered,
not whether the response was helpful — so that decision is also made in
Python rather than left to instruction-following.

These verdicts and rankings are injected on EVERY applicable query, including
when retrieval had to be relaxed. Suppressing them during relaxation was tried
and immediately regressed three eval cases — the scaffolding is what keeps the
model correct, and it is needed most when the retrieved set is imperfect.
"""

from .retriever import HybridRetriever
from .query_analyzer import analyze_query
from .generator import generate_answer

# NAAC grades ordered best → worst, for deterministic ranking
NAAC_ORDER = ["A++", "A+", "A", "B++", "B+", "B"]


class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()

    def answer(self, query: str, measure_cost: bool = False) -> dict:
        """
        End-to-end: query → analysis → retrieval → grounded answer.

        Returns JSON dict with keys: answer, citations, answered, reason_if_unanswered
        Optionally includes _metrics if measure_cost=True.
        """
        # Stage 1: Analyze the query
        analysis = analyze_query(query)

        # Stage 2: Retrieve relevant colleges
        # Even for full_scan, run through retrieve() so structured filters apply.
        docs = self.retriever.retrieve(query, top_k=analysis["top_k"])

        # Augment query with routing metadata for the LLM
        augmented_query = query
        notes = []

        # --- Over-constrained retrieval ---
        # No college satisfied every constraint, so the filter was relaxed and
        # these are nearest matches rather than valid answers. The correct
        # response depends on WHICH constraint failed: an unmatched course means
        # the data cannot answer at all and the system must refuse; an unmatched
        # budget or cutoff means near-misses are useful if the gap is named.
        relaxed = getattr(self.retriever, "last_retrieval_relaxed", False)
        filters = self.retriever.last_filters

        if relaxed:
            constraints = self.retriever.describe_filters()
            note = (
                "⚠️ NO EXACT MATCH: No college in the dataset satisfies all of "
                f"these constraints together — {constraints}. The records below are "
                "the closest available, NOT valid answers.\n"
            )
            if filters.get("courses"):
                note += (
                    "The unmet constraint includes a COURSE. If no college in the "
                    "records below actually offers the course the student asked for, "
                    "the correct response is a refusal: set answered=false and state "
                    "plainly that the course is not in the data. Do NOT list "
                    "near-miss colleges as though they answered the question.\n"
                )
            note += (
                "If the unmet constraint is numeric (budget, cutoff), you may present "
                "the closest options — but state explicitly which requirement each one "
                "fails and by how much."
            )
            notes.append(note)

        # --- Pre-computed budget / eligibility verdicts ---
        # Always injected when a numeric constraint exists, relaxed or not.
        if filters.get("max_fees_per_year") or filters.get("min_score"):
            verdict_lines = ["⚠️ PRE-COMPUTED FILTER RESULTS (trust these, do NOT re-compare):"]
            for doc in docs:
                row = doc["row"]
                cid = doc["college_id"]
                parts = []
                if filters.get("max_fees_per_year"):
                    budget = filters["max_fees_per_year"]
                    fee = row["annual_fees_inr"]
                    fits = "✅ WITHIN BUDGET" if fee <= budget else "❌ OVER BUDGET"
                    parts.append(f"Fee ₹{fee:,} vs budget ₹{budget:,} → {fits}")
                if filters.get("min_score"):
                    score = filters["min_score"]
                    cutoff = row["last_year_cutoff_pct"]
                    eligible = "✅ ELIGIBLE" if cutoff <= score else "❌ NOT ELIGIBLE"
                    parts.append(f"Cutoff {cutoff}% vs score {score}% → {eligible}")
                verdict_lines.append(f"  {cid}: {' | '.join(parts)}")
            notes.append("\n".join(verdict_lines))

        # --- Pre-computed rankings for subjective / comparison queries ---
        # Also always injected. This block is what excludes placement-zero
        # colleges from the placement ranking and labels them correctly.
        if analysis["is_subjective"] and len(docs) > 1:
            notes.append(self._build_rankings(docs))

        if analysis["has_unit_ambiguity"]:
            notes.append(f"⚠️ UNIT NOTE: {analysis['unit_note']}")

        if analysis["is_subjective"]:
            notes.append(
                "⚠️ SUBJECTIVITY NOTE: The student asked a subjective question. "
                "Do NOT declare a single 'best' college. Present the pre-computed "
                "rankings above, state which criterion each reflects, and let the "
                "student decide. Do NOT re-order these lists yourself."
            )

        if analysis["needs_about_field"]:
            notes.append(
                "ℹ️ This question likely requires information from the 'Details' / 'about' "
                "field of each college. Read those sections carefully."
            )

        if notes:
            augmented_query = query + "\n\n" + "\n".join(notes)

        # Stage 3: Generate grounded answer
        result = generate_answer(augmented_query, docs, measure_cost=measure_cost)

        # Stage 4: Deterministic correction of the `answered` flag.
        # Asked for a course no college offers, the model writes a correct
        # refusal — "No college offers B.Sc Agriculture, however C007 offers
        # B.Sc in other science streams" — and then sets answered=true, because
        # it produced text it judges useful. That judgement is about the
        # response; the flag is about the question. Decided here instead.
        result = self._correct_answered_flag(result, docs, filters, relaxed)

        return result

    def _correct_answered_flag(self, result: dict, docs: list[dict],
                               filters: dict, relaxed: bool) -> dict:
        """
        Override `answered` to False when the requested course is absent
        from every retrieved record.

        Only fires when retrieval was relaxed AND a course filter was set —
        i.e. the structured filter already established that nothing matched.
        Narrow by design: this must never flip a legitimate answer to False.
        """
        if not (relaxed and filters.get("courses")):
            return result

        offered_anywhere = any(
            any(course in doc["row"]["courses_offered"].lower()
                for course in filters["courses"])
            for doc in docs
        )

        if not offered_anywhere:
            result["answered"] = False
            if not result.get("reason_if_unanswered"):
                result["reason_if_unanswered"] = (
                    "Requested course is not offered by any college in the dataset"
                )

        return result

    def _build_rankings(self, docs: list[dict]) -> str:
        """
        Pre-sort retrieved colleges on each comparable dimension.

        The LLM receives finished ordered lists and is told not to re-order.
        This eliminates the class of error where an LLM writes an ordering
        that contradicts the numbers it just quoted — and it is also what
        keeps `avg_placement_lpa = 0` out of the placement ranking, since a
        college that reports nothing is not the worst performer.
        """
        lines = ["📊 PRE-COMPUTED RANKINGS (already sorted — reproduce as-is, do NOT re-order):"]

        # Cheapest first
        by_fee = sorted(docs, key=lambda d: d["row"]["annual_fees_inr"])
        fee_str = " < ".join(
            f"{d['college_id']} (₹{d['row']['annual_fees_inr']:,})" for d in by_fee
        )
        lines.append(f"  By annual fee, lowest to highest: {fee_str}")

        # Highest placement first, excluding not-reported
        reported = [d for d in docs if d["row"]["avg_placement_lpa"] > 0]
        not_reported = [d for d in docs if d["row"]["avg_placement_lpa"] == 0]
        if reported:
            by_placement = sorted(
                reported, key=lambda d: d["row"]["avg_placement_lpa"], reverse=True
            )
            plac_str = " > ".join(
                f"{d['college_id']} (₹{d['row']['avg_placement_lpa']} LPA)" for d in by_placement
            )
            lines.append(f"  By average placement, highest to lowest: {plac_str}")
        if not_reported:
            ids = ", ".join(d["college_id"] for d in not_reported)
            lines.append(
                f"  Placement not reported (excluded from ranking, NOT 'worst'): {ids}"
            )

        # Best NAAC grade first
        def naac_key(d):
            grade = d["row"]["naac_grade"]
            return NAAC_ORDER.index(grade) if grade in NAAC_ORDER else len(NAAC_ORDER)

        by_naac = sorted(docs, key=naac_key)
        naac_str = " ≥ ".join(
            f"{d['college_id']} ({d['row']['naac_grade']})" for d in by_naac
        )
        lines.append(f"  By NAAC grade, best to worst: {naac_str}")

        return "\n".join(lines)