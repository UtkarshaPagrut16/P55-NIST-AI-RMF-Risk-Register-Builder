"""U1: Lifecycle risk mapping.

The eight lifecycle stages describe WHEN in the project a risk arises.
They are deliberately separate from the NIST RMF functions (WHAT activity
addresses the risk), which live in rmf_classifier.py.
"""
import pandas as pd

from .config import LIFECYCLE_STAGES, LIFECYCLE_DESCRIPTIONS


def normalise_stage(value: str) -> str:
    """Map free text from the LLM (e.g. 'data prep') to one of the 8 stages."""
    if not isinstance(value, str):
        return LIFECYCLE_STAGES[0]
    text = value.strip().lower()
    for stage in LIFECYCLE_STAGES:
        if text == stage.lower():
            return stage
    aliases = {
        "plan": "Planning", "design": "Planning", "scoping": "Planning",
        "collect": "Data Collection", "acquisition": "Data Collection",
        "prep": "Data Preparation", "preprocess": "Data Preparation", "label": "Data Preparation",
        "develop": "Development", "training": "Development", "model": "Development",
        "evaluat": "Evaluation", "test": "Evaluation", "validat": "Evaluation",
        "deploy": "Deployment", "release": "Deployment", "operat": "Deployment",
        "monitor": "Monitoring", "maintenance": "Monitoring",
        "retire": "Retirement", "decommission": "Retirement", "sunset": "Retirement",
    }
    for key, stage in aliases.items():
        if key in text:
            return stage
    return LIFECYCLE_STAGES[0]


def lifecycle_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per stage: description, number of risks, top severity, highest score."""
    rows = []
    for stage in LIFECYCLE_STAGES:
        subset = df[df["Lifecycle Stage"] == stage] if not df.empty else df
        rows.append({
            "Stage": stage,
            "What happens": LIFECYCLE_DESCRIPTIONS[stage],
            "Risks": int(len(subset)),
            "Highest Score": int(subset["Risk Score"].max()) if len(subset) else 0,
            "Risk IDs": ", ".join(subset["Risk ID"].tolist()) if len(subset) else "-",
        })
    return pd.DataFrame(rows)


def stage_coverage_gaps(df: pd.DataFrame) -> list:
    """Stages that have no mapped risk - a useful completeness check."""
    if df.empty:
        return list(LIFECYCLE_STAGES)
    present = set(df["Lifecycle Stage"])
    return [s for s in LIFECYCLE_STAGES if s not in present]


def stage_severity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Stage x Severity count table used for the stacked chart."""
    from .config import SEVERITY_ORDER
    if df.empty:
        return pd.DataFrame(0, index=LIFECYCLE_STAGES, columns=SEVERITY_ORDER)
    table = pd.crosstab(df["Lifecycle Stage"], df["Severity"])
    table = table.reindex(index=LIFECYCLE_STAGES, columns=SEVERITY_ORDER, fill_value=0)
    return table
