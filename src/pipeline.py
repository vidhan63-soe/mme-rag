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
        # these are nearest matches rather than valid answers. Say so loudly —
        # presenting them as if they matched would be worse than refusing.
        relaxed = getattr(self.retriever, "last_retrieval_relaxed", False)
        if relaxed:
            constraints = self.retriever.describe_filters()
            notes.append(
                "⚠️ NO EXACT MATCH: No college in the dataset satisfies all of "
                f"these constraints together — {constraints}. The records below are "
                "the closest available, NOT valid answers.\n"
                "You MUST: (1) state plainly that nothing matches every requirement, "
                "(2) name which specific constraint(s) cannot be met, (3) then offer "
                "the closest alternatives and say exactly how each falls short. "
                "Do NOT present these as if they satisfied the student's criteria."
            )

        # --- Pre-computed budget / eligibility verdicts ---
        filters = self.retriever.last_filters
        if not relaxed and (filters.get("max_fees_per_year") or filters.get("min_score")):
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
        if analysis["is_subjective"] and len(docs) > 1 and not relaxed:
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

        return result

    def _build_rankings(self, docs: list[dict]) -> str:
        """
        Pre-sort retrieved colleges on each comparable dimension.

        The LLM receives finished ordered lists and is told not to re-order.
        This eliminates the class of error where an LLM writes an ordering
        that contradicts the numbers it just quoted.
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