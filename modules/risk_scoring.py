"""U4 (part 1): risk scoring.

Risk Score = Likelihood x Impact  (both on a 1-5 scale)
    1-4   -> Low
    5-9   -> Moderate
    10-16 -> High
    17-25 -> Critical
"""
import numpy as np
import pandas as pd

from .config import SEVERITY_BANDS, SEVERITY_ORDER


def clamp_scale(value, default=3) -> int:
    """Force a value onto the 1-5 integer scale (LLM output can be messy)."""
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(1, min(5, v))


def calculate_score(likelihood, impact) -> int:
    return clamp_scale(likelihood) * clamp_scale(impact)


def classify_severity(score) -> str:
    score = int(score)
    for low, high, label in SEVERITY_BANDS:
        if low <= score <= high:
            return label
    raise ValueError(f"Score {score} is outside the valid 1-25 range")


def apply_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute Likelihood, Impact, Risk Score and Severity for every row.

    Called after the LLM returns data and again after the user edits the table,
    so the score and severity can never drift out of sync with the inputs.
    """
    df = df.copy()
    if df.empty:
        return df
    df["Likelihood"] = df["Likelihood"].apply(clamp_scale)
    df["Impact"] = df["Impact"].apply(clamp_scale)
    df["Risk Score"] = df["Likelihood"] * df["Impact"]
    df["Severity"] = df["Risk Score"].apply(classify_severity)
    return df


def heatmap_matrix(df: pd.DataFrame) -> np.ndarray:
    """5x5 array of risk counts. Row index = Likelihood-1, column index = Impact-1."""
    grid = np.zeros((5, 5), dtype=int)
    for _, row in df.iterrows():
        grid[int(row["Likelihood"]) - 1, int(row["Impact"]) - 1] += 1
    return grid


def risk_statistics(df: pd.DataFrame) -> dict:
    """Headline numbers for the dashboard metric cards."""
    if df.empty:
        return {
            "total": 0, "average_score": 0.0, "max_score": 0,
            "by_severity": {s: 0 for s in SEVERITY_ORDER},
        }
    counts = df["Severity"].value_counts().to_dict()
    return {
        "total": int(len(df)),
        "average_score": round(float(df["Risk Score"].mean()), 2),
        "max_score": int(df["Risk Score"].max()),
        "by_severity": {s: int(counts.get(s, 0)) for s in SEVERITY_ORDER},
    }
