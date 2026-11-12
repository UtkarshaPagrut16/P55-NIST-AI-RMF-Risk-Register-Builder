"""Collect REAL results for the report: test outcomes, sample statistics and (if the
Sentence-BERT model can be loaded) genuine retrieval results.

Usage (from the application/ folder):
    python tools/collect_results.py ../documentation/results.json

Nothing here is invented: every number is computed or parsed at run time. If the model cannot
be loaded, the retrieval section records that fact instead of fake scores.
"""
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.nist_retriever import ModelUnavailableError, NISTRetriever, load_knowledge_base  # noqa: E402
from modules.rmf_classifier import classify_register, function_counts  # noqa: E402
from modules.risk_scoring import heatmap_matrix, risk_statistics  # noqa: E402
from modules.samples import load_sample, sample_register  # noqa: E402

out_path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "documentation" / "results.json")
results = {"generated_at": datetime.now().isoformat(timespec="seconds")}

# ---- 1. run pytest and parse the real outcome -------------------------------------------------
proc = subprocess.run([sys.executable, "-m", "pytest", "-v", "-rs", "--no-header", "-p", "no:cacheprovider"],
                      cwd=ROOT, capture_output=True, text=True)
text = proc.stdout
summary = re.findall(r"=+ (.*(?:passed|failed|skipped).*?) in ([\d.]+)s", text)
tests = []
for line in text.splitlines():
    m = re.match(r"(tests/\S+)::(\S+) (PASSED|FAILED|SKIPPED)", line)
    if m:
        tests.append({"file": m.group(1).split("/")[-1], "name": m.group(2), "status": m.group(3)})
skips = [l for l in text.splitlines() if l.startswith("SKIPPED")]
results["tests"] = {
    "summary": summary[-1][0] if summary else "unknown",
    "seconds": float(summary[-1][1]) if summary else None,
    "cases": tests,
    "skip_reasons": skips,
    "return_code": proc.returncode,
}

# ---- 2. sample statistics ----------------------------------------------------------------------
retriever, retr_error = None, None
try:
    retriever = NISTRetriever(use_cache=False)
except ModelUnavailableError as exc:
    retr_error = str(exc)

results["knowledge_base_entries"] = len(load_knowledge_base())
results["samples"] = {}
for key in ["healthcare_diagnosis", "recruitment_screening"]:
    df = sample_register(key)
    ret = retriever.retrieve_for_register(df, top_k=3) if retriever else None
    cl = classify_register(df, ret)
    entry = {
        "name": load_sample(key)["project"]["name"],
        "stats": risk_statistics(df),
        "heatmap": heatmap_matrix(df).tolist(),
        "rmf_counts": function_counts(cl),
        "register": df[["Risk ID", "Title", "Lifecycle Stage", "Risk Category", "Likelihood", "Impact",
                        "Risk Score", "Severity", "Responsible Stakeholder"]].to_dict("records"),
        "classification": cl[["Risk ID", "RMF Function", "RMF Subcategory"]].to_dict("records"),
        "retrieval": None,
    }
    if ret is not None:
        entry["retrieval"] = ret[["Risk ID", "Rank", "Guidance ID", "RMF Function", "Guidance Title", "Similarity"]].to_dict("records")
    results["samples"][key] = entry

results["sbert"] = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "ran": retriever is not None,
    "error": retr_error,
}

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("Saved", out_path)
print("Tests:", results["tests"]["summary"])
print("Sentence-BERT retrieval executed:", results["sbert"]["ran"])
