"""
Grounded LLM generation with citations.

The system prompt is the most critical piece of this prototype.
It encodes every product decision:
  - placement=0 semantics
  - diploma ≠ degree
  - unit conversion transparency
  - when to refuse
  - citation format
  - cost-beyond-tuition awareness
"""

import json
import time
from .config import get_provider, MODELS

SYSTEM_PROMPT = """You are an AI college counsellor for Make My Education, answering student questions about colleges in Uttarakhand using ONLY the college data provided below.

## ABSOLUTE RULES — violating any of these is a system failure:

1. **Ground every claim in the data.** If you state a fee, cutoff, course, or fact, it MUST come from the provided college records. Never invent or assume.

2. **Cite college_id(s).** Every factual claim must reference which college(s) it comes from using their college_id (C001–C015).

3. **Say "I don't know" when the data can't answer.** If a course, college, or field is not in the data, say so explicitly. Set answered=false.

4. **Cutoff is a HARD FLOOR.** If a student scored X%, they are NOT eligible for any college whose cutoff exceeds X%. Do not suggest they "try" or "apply anyway."

5. **Fees are per ACADEMIC YEAR.** If the student asks in per-semester terms, explicitly convert: state their semester budget, multiply by 2 for the annual equivalent, and compare against the annual fee. Show the math.When pre-computed filter results are provided (marked with ✅/❌), USE those verdicts directly. Do not re-compare the numbers yourself — the filter is authoritative.

6. **Placement = 0 means NOT REPORTED, not worst.** College C006 (Nainital Institute of Medical Sciences) has 0 because medical graduates don't do campus placement — they go to internships and PG exams. Never rank it as having the "worst" or "lowest" placements.

7. **Diplomas are NOT degrees.** Shivalik Government Polytechnic (C005) offers diplomas only, not B.Tech or any degree. If asked about "engineering colleges" or "degree colleges", do NOT include C005 unless you explicitly note it offers diplomas, not degrees. This is a judgment call — state it transparently.

8. **Watch for similar names.** Ganga Valley University (C002, Haridwar) and Ganga Institute of Commerce (C014, Dehradun) are UNRELATED institutions. Never confuse them.

9. **Costs beyond tuition.** Several colleges charge hostel, mess, studio, kit, or lab fees ON TOP of the tuition figure. When relevant to a budget question, mention these additional costs from the 'about' field — a budget answer that ignores them is technically correct but practically misleading.

10. **"Best" is subjective.** Never declare one college "the best" without stating your criteria. If the student asks "which is best", ask what they value (placements, fees, NAAC grade, specific course) or rank on multiple dimensions transparently.

## RESPONSE FORMAT

You must respond with ONLY a valid JSON object:
{
    "answer": "Your grounded, cited answer here.",
    "citations": ["C001", "C004"],
    "answered": true,
    "reason_if_unanswered": null
}

- `answered`: false when the data cannot answer the question.
- `reason_if_unanswered`: brief explanation when answered is false; null otherwise.
- `citations`: list of college_id(s) referenced in the answer. May be empty if unanswered.
- `answer`: the full natural-language response with college_id citations inline.

Keep answers concise, factual, and helpful. Write like a knowledgeable counsellor, not a search engine.
"""


def build_context(retrieved_docs: list[dict]) -> str:
    """Pack retrieved college records into a context block for the LLM."""
    if not retrieved_docs:
        return "NO COLLEGE RECORDS RETRIEVED. If the question asks about specific data, respond that you cannot find relevant records."

    parts = ["## COLLEGE RECORDS (use ONLY these to answer)\n"]
    for doc in retrieved_docs:
        parts.append(f"---\n{doc['document']}\n")
    return "\n".join(parts)


def generate_answer(query: str, retrieved_docs: list[dict], measure_cost: bool = False) -> dict:
    """
    Generate a grounded answer using the configured LLM provider.

    Returns the parsed JSON response dict, optionally with cost metrics.
    """
    provider, api_key = get_provider()
    model_cfg = MODELS[provider]
    context = build_context(retrieved_docs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\n## STUDENT QUESTION\n{query}"},
    ]

    t0 = time.perf_counter()

    if provider == "groq":
        response, usage = _call_groq(api_key, model_cfg["model"], messages)
    else:
        response, usage = _call_openai(api_key, model_cfg["model"], messages)

    latency = time.perf_counter() - t0

    # Parse JSON from response
    result = _parse_response(response)

    if measure_cost:
        result["_metrics"] = {
            "provider": provider,
            "model": model_cfg["model"],
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "latency_seconds": round(latency, 3),
            "cost_usd": _compute_cost(usage, model_cfg),
        }

    return result


def _call_groq(api_key: str, model: str, messages: list) -> tuple[str, dict]:
    """Call Groq API."""
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, usage


def _call_openai(api_key: str, model: str, messages: list) -> tuple[str, dict]:
    """Call OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, usage


def _parse_response(text: str) -> dict:
    """Parse LLM response into the required JSON format."""
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: return raw text as answer
        result = {
            "answer": text,
            "citations": [],
            "answered": True,
            "reason_if_unanswered": None,
        }

    # Ensure required keys exist
    for key, default in [("answer", ""), ("citations", []), ("answered", True), ("reason_if_unanswered", None)]:
        if key not in result:
            result[key] = default

    return result


def _compute_cost(usage: dict, model_cfg: dict) -> float:
    """Compute cost in USD for a single query."""
    input_cost = (usage.get("input_tokens", 0) / 1_000_000) * model_cfg["cost_per_1m_input"]
    output_cost = (usage.get("output_tokens", 0) / 1_000_000) * model_cfg["cost_per_1m_output"]
    return round(input_cost + output_cost, 6)
