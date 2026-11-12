"""U5: NIST AI RMF classification (GOVERN / MAP / MEASURE / MANAGE).

Two pieces of evidence are combined, and BOTH are shown to the user:

1. Rule-based baseline (always available):
   a table that maps each risk category + lifecycle stage to a primary RMF function
   and a relevant subcategory. The rationale is written out so it can be explained.
2. Semantic evidence (when Sentence-BERT retrieval has run):
   the RMF function of the best-matching NIST guidance entry (U3) is recorded as a
   second opinion. It does not silently override the baseline.

RMF functions describe WHAT KIND of risk-management activity addresses a risk.
Lifecycle stages (lifecycle.py) describe WHEN the risk occurs. They are kept in
separate columns and never mixed.
"""
import pandas as pd

from .config import RMF_FUNCTIONS, RMF_FUNCTION_SUMMARY

# category -> (primary function, subcategory id, subcategory label, explanation)
CATEGORY_RULES = {
    "Fairness and Bias": ("MEASURE", "MEASURE 2.11", "Fairness and bias evaluated",
                          "Bias is detected by measuring outcomes across groups, so the primary activity is MEASURE."),
    "Validity and Reliability": ("MEASURE", "MEASURE 2.5", "Valid and reliable",
                                 "Accuracy, robustness and generalisation are demonstrated through testing and monitoring (MEASURE)."),
    "Privacy": ("MEASURE", "MEASURE 2.10", "Privacy risk examined",
                "Privacy risk is examined and documented as part of evaluating the system (MEASURE)."),
    "Security and Resilience": ("MEASURE", "MEASURE 2.7", "Security and resilience evaluated",
                                "Resistance to attacks is evaluated and documented (MEASURE)."),
    "Transparency and Explainability": ("MEASURE", "MEASURE 2.9", "Explainability and interpretability",
                                        "Explanations are validated and documented as part of evaluation (MEASURE)."),
    "Safety": ("MEASURE", "MEASURE 2.6", "Safety evaluated",
               "Safety risks are evaluated regularly and failure behaviour is tested (MEASURE)."),
    "Accountability and Human Oversight": ("GOVERN", "GOVERN 3.2", "Human-AI oversight roles",
                                           "Who oversees and can override the AI is decided by policy and accountability structures (GOVERN)."),
    "Data Quality": ("MAP", "MAP 2.3", "Data representativeness and integrity",
                     "Data collection, selection and representativeness are identified when mapping the system (MAP)."),
    "Governance and Compliance": ("GOVERN", "GOVERN 1.1", "Legal and regulatory requirements",
                                  "Legal, policy and compliance obligations are established through governance (GOVERN)."),
}

# Lifecycle stages where the activity shifts toward a different function
STAGE_OVERRIDES = {
    ("Planning", "Fairness and Bias"): ("MAP", "MAP 1.1", "Intended purpose and context",
                                        "At planning time the risk is identified while framing context and impacts (MAP)."),
    ("Planning", "Privacy"): ("MAP", "MAP 1.1", "Intended purpose and context",
                              "Privacy obligations are identified while establishing context (MAP)."),
    ("Data Collection", "Fairness and Bias"): ("MAP", "MAP 2.3", "Data representativeness",
                                               "Bias in collected data is identified by mapping data representativeness (MAP)."),
    ("Data Collection", "Privacy"): ("MAP", "MAP 4.1", "Legal risks of data components",
                                     "Legal and privacy risks of data sources are mapped (MAP)."),
    ("Deployment", "Security and Resilience"): ("MANAGE", "MANAGE 2.3", "Respond to unknown risks",
                                                "At deployment the security response and recovery plan is the key control (MANAGE)."),
    ("Deployment", "Accountability and Human Oversight"): ("GOVERN", "GOVERN 3.2", "Human-AI oversight roles",
                                                           "Oversight roles for the deployed system are defined by policy (GOVERN)."),
    ("Monitoring", "Validity and Reliability"): ("MEASURE", "MEASURE 2.4", "Production monitoring",
                                                 "Drift and degradation are detected by monitoring the production system (MEASURE)."),
    ("Monitoring", "Fairness and Bias"): ("MEASURE", "MEASURE 2.11", "Fairness and bias evaluated",
                                          "Ongoing fairness is tracked with metrics after release (MEASURE)."),
    ("Monitoring", "Accountability and Human Oversight"): ("MANAGE", "MANAGE 4.3", "Incident communication",
                                                           "Incidents are tracked, communicated and fed into improvements (MANAGE)."),
    ("Retirement", "Privacy"): ("MANAGE", "MANAGE 2.4", "Supersede, disengage or deactivate",
                                "Safe decommissioning and data disposal are risk-treatment mechanisms (MANAGE)."),
    ("Retirement", "Governance and Compliance"): ("GOVERN", "GOVERN 1.7", "Decommissioning processes",
                                                  "Decommissioning processes are set by policy (GOVERN)."),
}

HIGH_SEVERITY = ("High", "Critical")


def classify_risk(category: str, stage: str, severity: str) -> dict:
    """Return the rule-based RMF classification for one risk."""
    rule = STAGE_OVERRIDES.get((stage, category)) or CATEGORY_RULES.get(category) or CATEGORY_RULES["Governance and Compliance"]
    function, sub_id, sub_label, why = rule
    out = {
        "RMF Function": function,
        "RMF Subcategory": sub_id,
        "Subcategory Meaning": sub_label,
        "Classification Rationale": why,
        "Treatment Function": "",
    }
    # high / critical risks also need a MANAGE response (prioritise + respond)
    if severity in HIGH_SEVERITY:
        out["Treatment Function"] = "MANAGE 1.2 / 1.3 (prioritise and respond to high-priority risk)"
    return out


def classify_register(df: pd.DataFrame, retrieval: pd.DataFrame = None) -> pd.DataFrame:
    """Add RMF columns to the register. `retrieval` is the output of NISTRetriever.retrieve_for_register."""
    if df.empty:
        return df.copy()
    rows = []
    for _, r in df.iterrows():
        c = classify_risk(r["Risk Category"], r["Lifecycle Stage"], r["Severity"])
        c["Risk ID"] = r["Risk ID"]
        c["Semantic Top Match"] = ""
        c["Semantic Function"] = ""
        c["Semantic Similarity"] = None
        if retrieval is not None and not retrieval.empty:
            top = retrieval[(retrieval["Risk ID"] == r["Risk ID"]) & (retrieval["Rank"] == 1)]
            if not top.empty:
                t = top.iloc[0]
                c["Semantic Top Match"] = f"{t['Guidance ID']}: {t['Guidance Title']}"
                c["Semantic Function"] = t["RMF Function"]
                c["Semantic Similarity"] = t["Similarity"]
        rows.append(c)
    return df.merge(pd.DataFrame(rows), on="Risk ID", how="left")


def function_counts(classified: pd.DataFrame) -> dict:
    if classified.empty:
        return {f: 0 for f in RMF_FUNCTIONS}
    counts = classified["RMF Function"].value_counts().to_dict()
    return {f: int(counts.get(f, 0)) for f in RMF_FUNCTIONS}


def function_explanations() -> dict:
    return dict(RMF_FUNCTION_SUMMARY)
