"""Tests for U3 (retrieval), U5 (RMF classification) and CSV export.

U3 has two layers of tests:
  * `StubEncoder` tests check ONLY the retrieval arithmetic (normalised vectors, cosine
    ranking, output columns). A stub is a test double - it proves nothing about semantic
    quality and is never used by the application.
  * `real_sbert` tests load the genuine all-MiniLM-L6-v2 model. They are SKIPPED automatically
    when the model cannot be downloaded (e.g. offline machine) - they are never faked.
"""
import io
import json

import numpy as np
import pandas as pd
import pytest

from modules.config import RISK_CATEGORIES, RMF_FUNCTIONS, SBERT_MODEL_NAME
from modules.export import register_to_csv
from modules.nist_retriever import (
    KB_PATH, ModelUnavailableError, NISTRetriever, load_knowledge_base, load_model,
)
from modules.rmf_classifier import CATEGORY_RULES, classify_register, classify_risk, function_counts
from modules.samples import sample_register


# ---------------------------------------------------------------- knowledge base
def test_knowledge_base_integrity():
    kb = load_knowledge_base()
    assert len(kb) >= 50
    ids = [e["id"] for e in kb]
    assert len(ids) == len(set(ids)), "duplicate guidance IDs"
    for e in kb:
        assert e["function"] in RMF_FUNCTIONS
        assert e["text"] and e["title"] and e["reference"].startswith("NIST AI")
    # every RMF function is represented
    assert {e["function"] for e in kb} == set(RMF_FUNCTIONS)
    meta = json.load(open(KB_PATH, encoding="utf-8"))["_meta"]
    assert "AI 100-1" in meta["primary_source"]


# ---------------------------------------------------------------- retrieval arithmetic (stub)
class StubEncoder:
    """Deterministic bag-of-characters encoder - TEST DOUBLE ONLY, checks plumbing not semantics."""
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for ch in t.lower():
                out[i, ord(ch) % 64] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


@pytest.fixture
def stub_retriever():
    return NISTRetriever(model=StubEncoder(), use_cache=False)


def test_retrieval_output_shape_and_order(stub_retriever):
    res = stub_retriever.search("privacy of patient data", top_k=4)
    assert len(res) == 4
    sims = [r["similarity"] for r in res]
    assert sims == sorted(sims, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in sims)
    assert {"id", "function", "guidance", "reference", "similarity"} <= set(res[0])


def test_exact_text_retrieves_itself_with_similarity_one(stub_retriever):
    entry = stub_retriever.entries[10]
    q = f"{entry['title']}. {entry['text']}"
    top = stub_retriever.search(q, top_k=1)[0]
    assert top["id"] == entry["id"] and top["similarity"] == pytest.approx(1.0, abs=1e-3)


def test_register_retrieval_table(stub_retriever):
    df = sample_register("healthcare_diagnosis")
    ret = stub_retriever.retrieve_for_register(df, top_k=3)
    assert len(ret) == len(df) * 3
    assert {"Risk ID", "Rank", "Guidance ID", "RMF Function", "Similarity", "Reference"} <= set(ret.columns)
    assert set(ret["RMF Function"]) <= set(RMF_FUNCTIONS)


# ---------------------------------------------------------------- real Sentence-BERT
@pytest.fixture(scope="module")
def real_sbert():
    try:
        return NISTRetriever(model=load_model(), use_cache=False)
    except ModelUnavailableError as exc:
        pytest.skip(f"Real Sentence-BERT model unavailable in this environment: {exc}")


def test_real_model_name_and_dimension(real_sbert):
    assert real_sbert.model_name == SBERT_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"
    assert real_sbert.embeddings.shape == (len(real_sbert.entries), 384)


def test_real_semantic_retrieval_beats_keywords(real_sbert):
    # query shares almost no words with the KB text but is about demographic fairness
    res = real_sbert.search("The model treats women and minority groups worse than others", top_k=5)
    assert any("Fair" in r["title"] or "bias" in r["title"].lower() for r in res)


def test_real_retrieval_for_every_sample_risk(real_sbert):
    for key in ["healthcare_diagnosis", "recruitment_screening"]:
        df = sample_register(key)
        ret = real_sbert.retrieve_for_register(df, top_k=3)
        assert len(ret) == len(df) * 3
        assert ret["Similarity"].between(0, 1).all()


def test_model_unavailable_raises_clear_error(monkeypatch):
    import sentence_transformers
    def boom(*a, **k):
        raise OSError("no internet")
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", boom)
    with pytest.raises(ModelUnavailableError):
        load_model()


# ---------------------------------------------------------------- U5 classification
def test_every_category_maps_to_valid_function():
    for cat in RISK_CATEGORIES:
        assert cat in CATEGORY_RULES
        assert CATEGORY_RULES[cat][0] in RMF_FUNCTIONS


@pytest.mark.parametrize("key", ["healthcare_diagnosis", "recruitment_screening"])
def test_classified_register_uses_all_four_functions_and_keeps_stage_separate(key):
    df = sample_register(key)
    cl = classify_register(df)
    assert len(cl) == len(df)
    assert set(cl["RMF Function"]) <= set(RMF_FUNCTIONS)
    assert set(cl["RMF Function"]) == set(RMF_FUNCTIONS)
    assert (cl["Lifecycle Stage"].values == df["Lifecycle Stage"].values).all()
    assert not set(cl["Lifecycle Stage"]) & set(RMF_FUNCTIONS)  # stages and functions never mixed
    assert cl["Classification Rationale"].str.len().min() > 20
    assert sum(function_counts(cl).values()) == len(df)


def test_critical_risk_gets_manage_treatment():
    c = classify_risk("Fairness and Bias", "Development", "Critical")
    assert "MANAGE" in c["Treatment Function"]
    assert classify_risk("Fairness and Bias", "Development", "Low")["Treatment Function"] == ""


def test_semantic_columns_added_when_retrieval_given(stub_retriever):
    df = sample_register("healthcare_diagnosis")
    ret = stub_retriever.retrieve_for_register(df)
    cl = classify_register(df, ret)
    assert (cl["Semantic Top Match"] != "").all()
    assert cl["Semantic Function"].isin(RMF_FUNCTIONS).all()


# ---------------------------------------------------------------- CSV export
def test_csv_roundtrip_contains_complete_register():
    df = sample_register("recruitment_screening")
    cl = classify_register(df)
    data = register_to_csv(df, cl)
    back = pd.read_csv(io.BytesIO(data))
    assert len(back) == len(df)
    for col in ["Risk ID", "Title", "Description", "Lifecycle Stage", "Risk Category", "Likelihood", "Impact",
                "Risk Score", "Severity", "Mitigation Strategy", "Responsible Stakeholder", "RMF Function"]:
        assert col in back.columns
    assert (back["Risk Score"] == back["Likelihood"] * back["Impact"]).all()
