"""Local fallback risk generator (used when no API key is set or the LLM call fails).

IMPORTANT: this is NOT an LLM. It selects from pre-written risk templates using simple
domain detection. Every row it produces is labelled "Local template fallback (not LLM)"
so the register can never be mistaken for LLM output.
"""
import pandas as pd

from .lifecycle import normalise_stage
from .llm_client import risks_to_dataframe

FALLBACK_LABEL = "Local template fallback (not LLM)"

DOMAIN_KEYWORDS = {
    "healthcare": ["health", "patient", "clinical", "diagnos", "hospital", "medical", "disease", "doctor"],
    "recruitment": ["recruit", "resume", "hiring", "candidate", "applicant", "job", "hr "],
}


def detect_domain(project: dict) -> str:
    """Very simple keyword check used ONLY to pick a template set (not part of U3)."""
    blob = " ".join(str(v) for v in project.values()).lower() + " "
    scores = {d: sum(k in blob for k in kws) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "generic"


def _generic_templates(name: str) -> list:
    n = name or "the AI system"

    def T(title, desc, stage, cat, L, I, mit, who):
        return dict(title=title, description=desc.replace("{n}", n), lifecycle_stage=stage,
                    risk_category=cat, likelihood=L, impact=I, mitigation=mit, stakeholder=who)

    return [
        T("Unclear purpose and success criteria", "The intended use, limits and success measures of {n} are not clearly defined, so teams may build or use it for unsuitable tasks.", "Planning", "Governance and Compliance", 3, 4, "Write an intended-use statement and risk-tolerance document; review with stakeholders.", "Project Manager"),
        T("Legal and regulatory non-compliance", "{n} may process personal data or make decisions that fall under privacy, consumer or sector regulations.", "Planning", "Governance and Compliance", 3, 5, "Perform a legal and data-protection review before development starts.", "Legal Counsel"),
        T("Unrepresentative or biased training data", "The data collected for {n} may not represent all user groups, causing unequal performance.", "Data Collection", "Fairness and Bias", 4, 4, "Audit coverage by group; broaden sources; document dataset limitations.", "Data Scientist"),
        T("Personal data collected without clear consent", "Data gathered for {n} may include personal information without a documented purpose or consent.", "Data Collection", "Privacy", 3, 4, "Apply data minimisation; record consent and purpose; restrict access.", "Data Protection Officer"),
        T("Poor data quality and labelling errors", "Missing values, noisy labels and inconsistent formats in the prepared data reduce the reliability of {n}.", "Data Preparation", "Data Quality", 4, 3, "Data validation checks; double labelling; documented cleaning steps.", "Data Engineer"),
        T("Proxy variables for sensitive attributes", "Features used by {n} may indirectly reveal protected attributes and lead to unfair outcomes.", "Data Preparation", "Fairness and Bias", 3, 4, "Proxy analysis and feature review before training.", "Data Scientist"),
        T("Overfitting and weak generalisation", "The model behind {n} may perform well in development but fail on real-world data.", "Development", "Validity and Reliability", 4, 3, "Cross-validation, regularisation and an untouched test set.", "ML Engineer"),
        T("Lack of explainability", "Users of {n} cannot understand why particular outputs were produced.", "Development", "Transparency and Explainability", 4, 3, "Add explanation methods and a model card.", "ML Engineer"),
        T("Insufficient testing for edge cases and subgroups", "Evaluation of {n} may only report averages and miss failures in rare or vulnerable cases.", "Evaluation", "Validity and Reliability", 3, 4, "Stress tests, subgroup metrics and independent review.", "QA Lead"),
        T("Undetected unfair outcomes", "Without fairness testing, {n} may produce biased results that are only discovered after harm occurs.", "Evaluation", "Fairness and Bias", 3, 4, "Compute fairness metrics across groups and document thresholds.", "Compliance Officer"),
        T("Security attacks on model and data", "{n} may be exposed to data poisoning, adversarial inputs, model theft or unauthorised access.", "Deployment", "Security and Resilience", 3, 5, "Threat modelling, access control, encryption and penetration testing.", "CISO"),
        T("Insufficient human oversight", "Decisions of {n} may be accepted without human review or an appeal route.", "Deployment", "Accountability and Human Oversight", 4, 4, "Define human-in-the-loop checkpoints, override and escalation procedures.", "Product Owner"),
        T("Model drift and performance decay", "Changes in real-world data cause the performance of {n} to degrade over time.", "Monitoring", "Validity and Reliability", 4, 3, "Monitor inputs and accuracy; alerts and scheduled retraining.", "ML Engineer"),
        T("Incidents not reported or learned from", "Failures or complaints about {n} are not logged, so the same problems recur.", "Monitoring", "Accountability and Human Oversight", 3, 3, "Incident register, user feedback channel and periodic review.", "Operations Manager"),
        T("Data and models left behind after retirement", "When {n} is switched off, personal data, models and logs may be retained without control.", "Retirement", "Privacy", 3, 3, "Retention and deletion schedule; secure disposal and confirmation.", "Data Protection Officer"),
        T("Loss of records and user disruption on retirement", "Withdrawing {n} abruptly can disrupt users and remove the audit trail.", "Retirement", "Governance and Compliance", 2, 3, "Phased retirement plan and archived audit logs.", "Project Manager"),
    ]


def generate_fallback_risks(project: dict, num_risks: int = 12) -> pd.DataFrame:
    """Build a register from templates. Healthcare/recruitment reuse the curated sample risks."""
    domain = detect_domain(project)
    if domain in ("healthcare", "recruitment"):
        from .samples import load_sample
        key = "healthcare_diagnosis" if domain == "healthcare" else "recruitment_screening"
        templates = load_sample(key)["risks"]
    else:
        templates = _generic_templates(project.get("name", ""))

    # keep at least one risk per lifecycle stage first, then fill by severity (L x I)
    ordered = sorted(templates, key=lambda r: -(r["likelihood"] * r["impact"]))
    chosen, seen_stages = [], set()
    for r in ordered:
        stage = normalise_stage(r["lifecycle_stage"])
        if stage not in seen_stages:
            chosen.append(r)
            seen_stages.add(stage)
    for r in ordered:
        if len(chosen) >= num_risks:
            break
        if r not in chosen:
            chosen.append(r)
    chosen = chosen[:max(num_risks, len(seen_stages))]
    chosen.sort(key=lambda r: templates.index(r))  # restore lifecycle order
    return risks_to_dataframe(chosen, FALLBACK_LABEL)
