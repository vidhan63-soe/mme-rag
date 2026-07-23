"""
Load sample_colleges.csv and build rich text documents for each college.

Design choice: each college becomes ONE document that merges structured fields
into readable prose alongside the free-text `about` field.  This lets embedding
search find colleges by ANY attribute — course, fees, location, hostel, or
scholarship info buried in `about` — without needing separate indexes.

For 15 records this is the pragmatic call.  At 10K+ colleges you'd split
structured filtering (SQL/pandas) from semantic search (embeddings on `about`)
and combine results in a retrieval fusion step.
"""

import pandas as pd
from .config import DATA_PATH


def load_dataframe(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV into a DataFrame with correct types."""
    df = pd.read_csv(path)
    df["annual_fees_inr"] = df["annual_fees_inr"].astype(int)
    df["last_year_cutoff_pct"] = df["last_year_cutoff_pct"].astype(int)
    df["total_seats"] = df["total_seats"].astype(int)
    df["avg_placement_lpa"] = df["avg_placement_lpa"].astype(float)
    df["established_year"] = df["established_year"].astype(int)
    return df


def build_college_document(row: pd.Series) -> str:
    """
    Create a rich, searchable text representation of one college.

    This is what gets embedded.  It's intentionally verbose so that
    semantic search can surface the college for queries about any field.
    """
    # Handle placement=0 correctly (not "worst", just not reported)
    if row["avg_placement_lpa"] == 0:
        placement_line = "Average placement package: Not reported / not applicable (see details below)."
    else:
        placement_line = f"Average placement package: ₹{row['avg_placement_lpa']} LPA."

    courses = row["courses_offered"].replace(";", ",")

    doc = f"""{row['name']} [{row['college_id']}]
Location: {row['city']}, {row['state']}
Type: {row['type']}
Courses offered: {courses}
Annual tuition fees: ₹{row['annual_fees_inr']:,} per academic year
Last year cutoff: {row['last_year_cutoff_pct']}% (hard minimum aggregate percentage)
Total seats: {row['total_seats']}
Hostel available: {row['hostel_available']}
NAAC grade: {row['naac_grade']}
{placement_line}
Established: {row['established_year']}

Details: {row['about']}"""
    return doc


def build_all_documents(df: pd.DataFrame = None) -> list[dict]:
    """
    Return a list of {"college_id", "name", "document", "row"} dicts.
    """
    if df is None:
        df = load_dataframe()
    docs = []
    for _, row in df.iterrows():
        docs.append({
            "college_id": row["college_id"],
            "name": row["name"],
            "document": build_college_document(row),
            "row": row.to_dict(),
        })
    return docs
