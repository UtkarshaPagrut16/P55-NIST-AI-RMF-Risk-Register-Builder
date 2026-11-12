"""Tests for U1 (lifecycle) and U4 (scoring)."""
import pandas as pd
import pytest

from modules.config import LIFECYCLE_STAGES
from modules.lifecycle import lifecycle_summary, normalise_stage, stage_coverage_gaps
from modules.risk_scoring import (
    apply_scoring, calculate_score, classify_severity, clamp_scale, heatmap_matrix, risk_statistics,
)
from modules.samples import sample_register


@pytest.mark.parametrize("score,expected", [
    (1, "Low"), (4, "Low"), (5, "Moderate"), (9, "Moderate"),
    (10, "High"), (16, "High"), (17, "Critical"), (25, "Critical"),
])
def test_severity_boundaries(score, expected):
    assert classify_severity(score) == expected


def test_every_l_x_i_combination_has_correct_score_and_severity():
    for L in range(1, 6):
        for I in range(1, 6):
            s = calculate_score(L, I)
            assert s == L * I
            sev = classify_severity(s)
            if s <= 4: assert sev == "Low"
            elif s <= 9: assert sev == "Moderate"
            elif s <= 16: assert sev == "High"
            else: assert sev == "Critical"


def test_out_of_range_score_rejected():
    with pytest.raises(ValueError):
        classify_severity(26)
    with pytest.raises(ValueError):
        classify_severity(0)


@pytest.mark.parametrize("raw,expected", [(0, 1), (9, 5), ("4", 4), (None, 3), ("abc", 3), (2.6, 3)])
def test_clamp_scale(raw, expected):
    assert clamp_scale(raw) == expected


def test_apply_scoring_recomputes_after_edit():
    df = pd.DataFrame({"Likelihood": [5, 1], "Impact": [5, 2], "Risk Score": [0, 0], "Severity": ["", ""]})
    out = apply_scoring(df)
    assert out["Risk Score"].tolist() == [25, 2]
    assert out["Severity"].tolist() == ["Critical", "Low"]


def test_heatmap_matrix_counts_match_register():
    df = sample_register("healthcare_diagnosis")
    grid = heatmap_matrix(df)
    assert grid.shape == (5, 5)
    assert grid.sum() == len(df)


def test_statistics_add_up():
    df = sample_register("recruitment_screening")
    st = risk_statistics(df)
    assert st["total"] == len(df) == sum(st["by_severity"].values())


def test_normalise_stage():
    assert normalise_stage("data prep") == "Data Preparation"
    assert normalise_stage("DEPLOYMENT") == "Deployment"
    assert normalise_stage("decommissioning") == "Retirement"
    assert normalise_stage(None) == "Planning"


@pytest.mark.parametrize("key", ["healthcare_diagnosis", "recruitment_screening"])
def test_samples_cover_all_eight_lifecycle_stages(key):
    df = sample_register(key)
    assert stage_coverage_gaps(df) == []
    summary = lifecycle_summary(df)
    assert summary["Stage"].tolist() == LIFECYCLE_STAGES
    assert summary["Risks"].sum() == len(df)
