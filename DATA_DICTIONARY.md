# Data Dictionary — `sample_colleges.csv`

15 synthetic colleges. Values are invented for this exercise; treat them as if they were real and verified.
**Read this file before you start.** Several evaluation questions turn on the semantics below, not on the numbers.

| Column | Type | Meaning |
| --- | --- | --- |
| `college_id` | string | Stable identifier, `C001`–`C015`. **Use this in your citations.** |
| `name` | string | Full college name. Two entries have deliberately similar names. |
| `city` | string | City |
| `state` | string | State (all Uttarakhand in this sample) |
| `type` | enum | `Government` \| `Private` \| `Deemed` — **three** values, not two |
| `courses_offered` | string | Semicolon-separated. A `Diploma` is not a degree. |
| `annual_fees_inr` | integer | Tuition in ₹ **per academic year**. Not per semester. Excludes hostel, mess and any charges described in `about`. |
| `last_year_cutoff_pct` | integer | Minimum aggregate percentage admitted last year. **Treat as a hard floor** — a student below this figure was not eligible. |
| `total_seats` | integer | Total intake across all courses |
| `hostel_available` | boolean | `Yes` \| `No` |
| `naac_grade` | string | `A++`, `A+`, `A`, `B++`, `B+`, `B` |
| `avg_placement_lpa` | float | Average package, lakhs per annum. **`0` means not reported / not applicable.** `0` does **not** mean "worst placements" — see `C006`. |
| `established_year` | integer | |
| `about` | free text | ~110 words per college: admission process, scholarships, hostel arrangements, extra charges, placement context, faculty. **Unstructured. Some questions can only be answered from this field, and it sometimes qualifies or explains a structured value.** |

---

## Notes that will save you time

**Units.** Fees are per **year**. Students speak in **per semester**, in **lakhs**, and in **total course cost**. Your system will receive all of these. Decide how to handle the ambiguity — convert, clarify, or state your assumption explicitly — and say what you chose in the README. Silently answering a per-semester question with a per-year figure is the worst failure available in this exercise.

**Cutoff.** A hard minimum, expressed as an aggregate percentage. Not a percentile, not a rank.

**Placement `0`.** Exactly one college reports `0`. Its `about` field explains why. A system that calls it the worst-performing college has not understood its data.

**Diplomas.** One institution awards diplomas only, not degrees. Whether it belongs in an answer about "engineering colleges" is a judgment call. Make one, and be able to defend it.

**Costs beyond tuition.** Several `about` fields describe hostel, mess, studio, kit or laboratory charges levied over and above `annual_fees_inr`. A budget answer that ignores these is technically grounded and practically wrong. How you handle that is a product decision, and we're interested in it.

**Similar names.** Two colleges share a leading word and are unrelated institutions in different cities. Retrieval will confuse them if you let it.

**Nothing outside this file.** The dataset is the whole world. If a question asks for something not present — a field, a course, a college — the correct answer is to say so.
