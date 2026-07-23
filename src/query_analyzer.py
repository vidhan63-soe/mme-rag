"""
Query analysis and routing.

Determines whether a query needs:
  - Full scan (comparison, listing, "all colleges that...")
  - Targeted retrieval (specific college, specific filter)
  - Semantic-only search (scholarship, about-field questions)
  - Refusal (asks for data outside the dataset)

This avoids unnecessary embedding lookups for simple structured queries
and ensures semantic queries aren't missed by structured filters.
"""

import re


def analyze_query(query: str) -> dict:
    """
    Analyze a query and return routing metadata.

    Returns:
        {
            "strategy": "full_scan" | "targeted" | "semantic" | "hybrid",
            "needs_about_field": bool,
            "has_unit_ambiguity": bool,
            "unit_note": str | None,
            "is_subjective": bool,
            "top_k": int,
        }
    """
    q = query.lower()
    result = {
        "strategy": "hybrid",
        "needs_about_field": False,
        "has_unit_ambiguity": False,
        "unit_note": None,
        "is_subjective": False,
        "top_k": 5,
    }

    # --- Full scan triggers ---
    full_scan_patterns = [
        r"\blist\b.*\bcollege",
        r"\ball\b.*\bcollege",
        r"\bwhich colleges\b",
        r"\bcompare\b",
        r"\bhow many\b",
        r"\beveryone\b|\bevery college\b",
    ]
    for pat in full_scan_patterns:
        if re.search(pat, q):
            result["strategy"] = "full_scan"
            result["top_k"] = 15
            break

    # --- Specific college → targeted ---
    specific_indicators = [
        "what's the", "what is the", "does .* offer", "tell me about",
        "average placement", "cutoff at", "fees at", "about .*college",
        "about .*university", "about .*institute", "about .*school",
    ]
    for pat in specific_indicators:
        if re.search(pat, q):
            result["strategy"] = "targeted"
            result["top_k"] = 3
            break

    # --- Semantic-only (about field) ---
    semantic_keywords = [
        "scholarship", "financial aid", "fee concession", "fee waiver",
        "low-income", "income", "faculty", "placement cell",
        "internship", "hostel facility", "hostel facilities",
        "campus life", "lab", "workshop", "research",
    ]
    if any(kw in q for kw in semantic_keywords):
        result["needs_about_field"] = True
        if result["strategy"] != "full_scan":
            result["strategy"] = "semantic"
            result["top_k"] = 10

    # --- Unit ambiguity detection ---
    # Students state budgets in three incompatible units: per semester,
    # per year (our data's unit), and total for the whole course. Silently
    # answering one with another is the worst failure available here.
    #
    # Crucially, this must distinguish a student STATING THEIR BUDGET
    # ("my total budget is 5 lakhs") from a student ASKING WHAT SOMETHING
    # COSTS ("what's the total cost of studying at X?"). Both contain the
    # phrase "total cost", but only the first needs a conversion. Firing on
    # the second injects a divide-by-duration instruction into a plain
    # lookup and crowds out the actual answer — which is exactly what
    # happened to the hidden-costs eval case before this guard was added.
    money_words = ["budget", "afford", "cost", "fee", "lakh", "rupee", "₹"]
    total_words = ["total budget", "total cost", "overall budget",
                   "entire course", "full course", "whole course", "total fee"]
    possessive_signals = ["my ", "i have", "i can", "i've got", "budget of",
                          "afford", "for me"]

    if ("semester" in q or "sem " in q) and any(w in q for w in money_words):
        result["has_unit_ambiguity"] = True
        result["unit_note"] = (
            "The student's budget is stated per SEMESTER. "
            "Our fee data is per ACADEMIC YEAR (2 semesters). "
            "Convert: semester_budget × 2 = annual budget for comparison, "
            "and show that multiplication in your answer."
        )
    elif (any(w in q for w in total_words)
          and any(w in q for w in possessive_signals)):
        years_match = re.search(r'(\d+)\s*[-\s]?year', q)
        years = years_match.group(1) if years_match else "the course duration"
        result["has_unit_ambiguity"] = True
        result["unit_note"] = (
            "The student's budget is stated as a TOTAL for the whole course "
            f"({years} years). Our fee data is per ACADEMIC YEAR. "
            "Convert: total_budget ÷ number_of_years = annual budget, "
            "and show that division explicitly in your answer."
        )

    # --- Subjectivity detection ---
    subjective_words = ["best", "top", "recommend", "should i", "which one",
                        "better", "ideal", "good for me", "right for me",
                        "suggest", "worst", "lowest", "highest"]
    if any(w in q for w in subjective_words):
        result["is_subjective"] = True

    return result