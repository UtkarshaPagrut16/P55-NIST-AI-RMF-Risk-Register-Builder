"""U3: Sentence-BERT semantic retrieval over a local NIST AI RMF knowledge base.

How it works (easy version for viva):
1. Every guidance entry in data/nist_ai_rmf_knowledge_base.json is turned into a
   384-dimensional vector by the Sentence-BERT model all-MiniLM-L6-v2.
2. Each generated risk (title + description) is turned into a vector the same way.
3. Because the vectors are L2-normalised, cosine similarity = dot product.
   The entries with the highest similarity are the retrieved guidance.

There is intentionally NO keyword matching here. If the model cannot be loaded
(for example no internet on first run) the retriever raises ModelUnavailableError
and the UI tells the user - it never pretends to have done semantic search.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SBERT_MODEL_NAME, DEFAULT_TOP_K

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KB_PATH = DATA_DIR / "nist_ai_rmf_knowledge_base.json"
CACHE_DIR = DATA_DIR / "cache"


class ModelUnavailableError(RuntimeError):
    """Raised when the Sentence-BERT model cannot be loaded."""


def load_knowledge_base(path: Path = KB_PATH) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["entries"]


def kb_text(entry: dict) -> str:
    """Text that is embedded for a knowledge-base entry."""
    return f"{entry['title']}. {entry['text']}"


def risk_text(title: str, description: str) -> str:
    """Text that is embedded for a generated risk."""
    return f"{title}. {description}"


def load_model(model_name: str = SBERT_MODEL_NAME):
    """Load Sentence-BERT. Raises ModelUnavailableError with a clear message on failure."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as exc:  # network error, missing package, corrupt cache ...
        raise ModelUnavailableError(
            f"Could not load Sentence-BERT model '{model_name}': {type(exc).__name__}: {exc}"
        ) from exc


class NISTRetriever:
    def __init__(self, model=None, kb_path: Path = KB_PATH, use_cache: bool = True):
        self.entries = load_knowledge_base(kb_path)
        self.model = model if model is not None else load_model()
        self.model_name = SBERT_MODEL_NAME
        self.embeddings = self._kb_embeddings(use_cache)

    # -- embeddings -------------------------------------------------------
    def _encode(self, texts: list) -> np.ndarray:
        return self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )

    def _kb_embeddings(self, use_cache: bool) -> np.ndarray:
        texts = [kb_text(e) for e in self.entries]
        digest = hashlib.md5(("|".join(texts) + self.model_name).encode("utf-8")).hexdigest()[:12]
        cache_file = CACHE_DIR / f"kb_embeddings_{digest}.npy"
        if use_cache and cache_file.exists():
            return np.load(cache_file)
        emb = self._encode(texts)
        if use_cache:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, emb)
            except OSError:
                pass  # read-only file system (e.g. cloud) - cache is optional
        return emb

    # -- search -----------------------------------------------------------
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list:
        q = self._encode([query])[0]
        sims = self.embeddings @ q  # cosine similarity (vectors are normalised)
        order = np.argsort(-sims)[:top_k]
        results = []
        for rank, idx in enumerate(order, start=1):
            e = self.entries[int(idx)]
            results.append({
                "rank": rank,
                "id": e["id"],
                "function": e["function"],
                "category": e["category"],
                "title": e["title"],
                "guidance": e["text"],
                "reference": e["reference"],
                "similarity": round(float(sims[idx]), 4),
            })
        return results

    def retrieve_for_register(self, df: pd.DataFrame, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
        """Retrieve top-k guidance for every risk. Returns a long table (one row per match)."""
        if df.empty:
            return pd.DataFrame()
        queries = [risk_text(r["Title"], r["Description"]) for _, r in df.iterrows()]
        q_emb = self._encode(queries)
        sims = q_emb @ self.embeddings.T
        rows = []
        for i, (_, risk) in enumerate(df.iterrows()):
            for rank, idx in enumerate(np.argsort(-sims[i])[:top_k], start=1):
                e = self.entries[int(idx)]
                rows.append({
                    "Risk ID": risk["Risk ID"],
                    "Risk Title": risk["Title"],
                    "Rank": rank,
                    "Guidance ID": e["id"],
                    "RMF Function": e["function"],
                    "RMF Category": e["category"],
                    "Guidance Title": e["title"],
                    "Guidance": e["text"],
                    "Similarity": round(float(sims[i, idx]), 4),
                    "Reference": e["reference"],
                })
        return pd.DataFrame(rows)
