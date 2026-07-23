"""
Hybrid retriever: structured filtering + semantic embedding search.

Strategy
--------
1.  **Structured filter** — extract numeric / categorical constraints from the
    query (budget, cutoff score, course, college type, hostel, specific college
    name) using rule-based extraction.  Apply as hard filters on the DataFrame.

2.  **Semantic search** — embed the query with the same model used for college
    documents and rank by cosine similarity.  This catches scholarship questions,
    about-field nuances, and anything the structured filter can't parse.

3.  **Combine** —
    • If structured filters matched specific colleges → use those, ranked by
      semantic similarity.
    • If no structured filters fired → return top-K by semantic similarity.
    • If filters over-constrained to ZERO results → fall back to semantic-only
      and set `last_retrieval_relaxed`, so the caller can tell the model that
      no record satisfies every constraint.
    • A college named explicitly in the query is always included, even if other
      filters would exclude it — the LLM needs its record to answer "no".

Why not just semantic search?
    "Budget ₹1.5 lakh" → cosine similarity has no notion of ≤.  A college
    costing ₹8.5 lakh might embed near the query because the word "lakh"
    appears.  Structured filtering is exact.

Why not just structured filtering?
    "Which colleges offer scholarships for low-income families?" has zero
    structured fields.  The answer lives in free text.

Why the empty-set fallback matters
    "I got 90% in Arts, which B.Tech is good for me?" intersects a course
    filter with a score filter and returns nothing.  Reporting "no records
    found" is technically true and practically useless — the student needs to
    hear *why* nothing matched.  Falling back to semantic retrieval gives the
    model something concrete to explain the gap with.

Known limitation
----------------
Negation is handled explicitly for hostel only.  Other negated constraints
("colleges NOT in Dehradun", "not government") are not detected and will be
filtered as though the negation were absent.  See README.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL
from .data_loader import load_dataframe, build_all_documents


class HybridRetriever:
    def __init__(self):
        self.df = load_dataframe()
        self.docs = build_all_documents(self.df)
        self.model = SentenceTransformer(EMBED_MODEL)
        # Pre-compute embeddings for all college documents
        texts = [d["document"] for d in self.docs]
        self.doc_embeddings = self.model.encode(texts, normalize_embeddings=True)
        # Set by retrieve() when structured filters had to be relaxed
        self.last_retrieval_relaxed = False
        self.last_filters = {}

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Return up to top_k college records relevant to the query.

        Each result is a dict with keys:
            college_id, name, document, row, similarity_score

        Side effects: sets `last_retrieval_relaxed` and `last_filters`, which
        pipeline.py reads to decide what to tell the model.
        """
        # --- Stage 1: structured filtering ---
        filters = self._extract_filters(query)
        self.last_filters = filters
        self.last_retrieval_relaxed = False

        mask = np.ones(len(self.df), dtype=bool)

        if filters.get("max_fees_per_year"):
            mask &= self.df["annual_fees_inr"].values <= filters["max_fees_per_year"]

        if filters.get("min_score"):
            # Student's score must meet or exceed the cutoff
            mask &= self.df["last_year_cutoff_pct"].values <= filters["min_score"]

        if filters.get("courses"):
            course_mask = np.zeros(len(self.df), dtype=bool)
            for _, row in self.df.iterrows():
                offered = row["courses_offered"].lower()
                for c in filters["courses"]:
                    if c in offered:
                        course_mask[row.name] = True
                        break
            mask &= course_mask

        if filters.get("college_type"):
            type_mask = self.df["type"].str.lower().values == filters["college_type"]
            mask &= type_mask

        # Hostel: three states — required (True), explicitly absent (False),
        # or not mentioned (key missing).  `.get()` would collapse False into
        # the missing case, so test membership instead.
        if "hostel_required" in filters:
            want = "yes" if filters["hostel_required"] else "no"
            mask &= self.df["hostel_available"].str.lower().values == want

        # --- Named college handling ---
        # When a specific college is named, ALWAYS include it so the LLM
        # can see its data and make a grounding decision (e.g., "Does X offer PhD?"
        # needs X's record to say "no, they don't").
        named_indices = set()
        if filters.get("college_name"):
            target = filters["college_name"]
            name_mask = np.zeros(len(self.df), dtype=bool)
            for idx, doc in enumerate(self.docs):
                if target in doc["name"].lower():
                    name_mask[idx] = True
                    named_indices.add(idx)
            mask &= name_mask

        # --- Stage 2: semantic similarity ---
        q_embedding = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.doc_embeddings, q_embedding.T).flatten()

        # --- Stage 3: combine ---
        structured_hit = mask.sum() < len(self.df)  # filters actually narrowed results

        if structured_hit and mask.sum() == 0 and not named_indices:
            # Filters over-constrained to nothing.  Returning an empty list makes
            # the model say "no records found", which is true but unhelpful — the
            # student learns nothing about WHY nothing matched.  Fall back to
            # semantic retrieval and flag it so pipeline.py can tell the model to
            # name the unmet constraints explicitly.
            self.last_retrieval_relaxed = True
            ranked = np.argsort(-similarities)[:top_k]
            candidates = [(idx, similarities[idx]) for idx in ranked]

        elif structured_hit:
            # Use filtered set, ranked by similarity
            candidates = []
            for idx in range(len(self.docs)):
                if mask[idx]:
                    candidates.append((idx, similarities[idx]))
            # If named college was filtered out by OTHER filters (e.g., course),
            # still include it — the LLM needs to see it to answer "no"
            for idx in named_indices:
                if not mask[idx]:
                    candidates.append((idx, similarities[idx]))
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:top_k]

        else:
            # Pure semantic — return top_k by similarity
            ranked = np.argsort(-similarities)[:top_k]
            candidates = [(idx, similarities[idx]) for idx in ranked]

        results = []
        for idx, sim in candidates:
            entry = dict(self.docs[idx])
            entry["similarity_score"] = float(sim)
            results.append(entry)

        return results

    def get_all_colleges(self) -> list[dict]:
        """Return all college docs (for queries needing full scan)."""
        return self.docs

    def describe_filters(self, filters: dict = None) -> str:
        """
        Human-readable summary of the constraints that were applied.

        Used when retrieval was relaxed, so the model can name exactly which
        constraints no college satisfied.
        """
        f = filters if filters is not None else self.last_filters
        parts = []
        if f.get("max_fees_per_year"):
            parts.append(f"annual fee at or below ₹{f['max_fees_per_year']:,}")
        if f.get("min_score"):
            parts.append(f"cutoff at or below {f['min_score']}%")
        if f.get("courses"):
            parts.append(f"offers one of: {', '.join(sorted(f['courses']))}")
        if f.get("college_type"):
            parts.append(f"type is {f['college_type'].title()}")
        if "hostel_required" in f:
            parts.append("has a hostel" if f["hostel_required"] else "has no hostel")
        if f.get("college_name"):
            parts.append(f"name matches '{f['college_name']}'")
        return "; ".join(parts) if parts else "no structured constraints"

    def _extract_filters(self, query: str) -> dict:
        """
        Rule-based extraction of structured constraints from natural language.

        Handles:
        - Budget in ₹, lakhs, per year / per semester
        - Score / percentage
        - Course keywords
        - College type (government/private/deemed)
        - Hostel requirement, including negation
        - Specific college name
        """
        q = query.lower()
        filters = {}

        # --- Budget extraction ---
        budget = self._extract_budget(q)
        if budget:
            filters["max_fees_per_year"] = budget

        # --- Score / percentage ---
        score_match = re.search(r'(\d+)\s*%', q)
        if score_match:
            # Only treat as a student score if context suggests it
            score_val = int(score_match.group(1))
            score_context_words = ["scored", "score", "marks", "percentage", "aggregate",
                                   "got", "have", "my", "i "]
            if any(w in q for w in score_context_words):
                filters["min_score"] = score_val

        # --- Course extraction ---
        course_map = {
            "engineering": ["b.tech", "b.arch", "m.tech"],
            "mba": ["mba"],
            "bba": ["bba"],
            "bcom": ["b.com"],
            "b.com": ["b.com"],
            "mcom": ["m.com"],
            "m.com": ["m.com"],
            "law": ["llb", "ba-llb", "llm"],
            "medical": ["mbbs", "bds"],
            "nursing": ["nursing"],
            "pharmacy": ["pharm"],
            "design": ["b.des", "b.f.a", "m.des"],
            "hotel management": ["bhm", "hospitality"],
            "media": ["bjmc", "ba-film", "mass comm"],
            "mca": ["mca"],
            "diploma": ["diploma"],
            "b.tech": ["b.tech"],
            "phd": ["phd", "doctoral"],
            "bca": ["bca"],
            "ca foundation": ["ca-foundation", "ca foundation"],
            "ca-foundation": ["ca-foundation", "ca foundation"],
        }
        found_courses = []
        for keyword, course_terms in course_map.items():
            # Word boundary avoids "ca" matching "can", etc.
            if re.search(r'\b' + re.escape(keyword) + r'\b', q):
                found_courses.extend(course_terms)
        if found_courses:
            filters["courses"] = list(set(found_courses))

        # --- College type ---
        if "government" in q or "govt" in q or "sarkari" in q:
            filters["college_type"] = "government"
        elif "private" in q:
            filters["college_type"] = "private"
        elif "deemed" in q:
            filters["college_type"] = "deemed"

        # --- Hostel (negation-aware) ---
        # "colleges with a hostel" and "colleges without a hostel" both contain
        # the token "hostel".  Treating them identically returns the exact
        # opposite set and produces a confident wrong answer, so check for
        # negation before setting the flag.
        if "hostel" in q:
            negation_words = [
                "without", "no hostel", "don't have", "dont have",
                "doesn't have", "doesnt have", "does not have",
                "lack", "not have", "no residential", "lacking",
                "absent", "unavailable",
            ]
            if any(w in q for w in negation_words):
                filters["hostel_required"] = False
            else:
                filters["hostel_required"] = True

        # --- Specific college name ---
        college_names = [
            "north ridge institute of technology",
            "ganga valley university",
            "himalayan college of engineering",
            "doon business school",
            "shivalik government polytechnic",
            "nainital institute of medical sciences",
            "kumaon arts and science college",
            "rishikesh institute of design",
            "terai technical university",
            "mussoorie college of hotel management",
            "haldwani law college",
            "ambedkar national institute",
            "char dham pharmacy college",
            "ganga institute of commerce",
            "silver peak school of media",
        ]
        for name in college_names:
            if name in q:
                filters["college_name"] = name
                break
        # Also try partial match on distinctive words
        if "college_name" not in filters:
            partial_names = {
                "north ridge": "north ridge institute of technology",
                "ganga valley": "ganga valley university",
                "himalayan college": "himalayan college of engineering",
                "doon business": "doon business school",
                "shivalik": "shivalik government polytechnic",
                "nainital institute": "nainital institute of medical sciences",
                "kumaon": "kumaon arts and science college",
                "rishikesh institute": "rishikesh institute of design",
                "terai technical": "terai technical university",
                "mussoorie college": "mussoorie college of hotel management",
                "haldwani law": "haldwani law college",
                "ambedkar national": "ambedkar national institute",
                "char dham": "char dham pharmacy college",
                "ganga institute": "ganga institute of commerce",
                "silver peak": "silver peak school of media",
            }
            for partial, full in partial_names.items():
                if partial in q:
                    filters["college_name"] = full
                    break

        return filters

    def _extract_budget(self, q: str) -> int | None:
        """
        Parse budget from query, handling:
        - "₹1.5 lakh/year" → 150000
        - "1 lakh per semester" → 200000 (×2 for annual)
        - "₹50,000" → 50000
        - "budget of 2 lakhs" → 200000
        """
        is_semester = "semester" in q or "sem" in q

        # Pattern: X lakh(s)
        lakh_match = re.search(r'[₹rs.\s]*(\d+\.?\d*)\s*(?:lakh|lac|l)(?:s)?', q)
        if lakh_match:
            amount = float(lakh_match.group(1)) * 100000
            if is_semester:
                amount *= 2  # convert semester to annual
            return int(amount)

        # Pattern: ₹X,XX,XXX or Rs X,XX,XXX
        rupee_match = re.search(r'[₹rs.\s]*(\d[\d,]*\d)', q)
        if rupee_match:
            amount_str = rupee_match.group(1).replace(",", "")
            if amount_str.isdigit():
                amount = int(amount_str)
                if is_semester:
                    amount *= 2
                return amount

        return None