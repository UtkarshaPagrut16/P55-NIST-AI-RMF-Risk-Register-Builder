"""Sample projects (Project A healthcare, Project B recruitment).

The risk registers in data/samples/*.json are hand-authored by the project author.
They are labelled 'Curated sample' - they are NOT produced by an LLM.
"""
import json
from pathlib import Path

import pandas as pd

from .llm_client import risks_to_dataframe

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
SAMPLE_LABEL = "Curated sample (hand-authored)"

SAMPLE_CHOICES = {
    "Project A: Healthcare Diagnosis System": "healthcare_diagnosis",
    "Project B: Recruitment & Resume Screening": "recruitment_screening",
}


def load_sample(key: str) -> dict:
    with open(SAMPLES_DIR / f"{key}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def sample_register(key: str) -> pd.DataFrame:
    return risks_to_dataframe(load_sample(key)["risks"], SAMPLE_LABEL)
