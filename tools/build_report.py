"""Build the academic PDF report from REAL project data.

Usage (from the application/ folder, after running collect_results.py and capture_screenshots.py):
    python tools/build_report.py

Inputs : ../documentation/results.json, ../screenshots/*.png
Output : ../55_Utkarsha_Pagrut_P55.pdf

The text adapts to what really happened: if Sentence-BERT could not run when results.json was
collected, the report says so; if it ran, the real retrieval results are included.
"""
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT.parent
sys.path.insert(0, str(ROOT))
from modules.config import SEVERITY_COLORS, RMF_COLORS, STUDENT  # noqa: E402

RES = json.loads((SUB / "documentation" / "results.json").read_text(encoding="utf-8"))
SHOTS = SUB / "screenshots"
OUT = SUB / "55_Utkarsha_Pagrut_P55.pdf"
TMP = Path(tempfile.mkdtemp())
SBERT_RAN = RES["sbert"]["ran"]
NAVY, INDIGO = colors.HexColor("#1E2A52"), colors.HexColor("#5B5BD6")

# ------------------------------------------------------------------ styles
ss = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontName="Times-Roman", fontSize=11, leading=15.5, alignment=TA_JUSTIFY, spaceAfter=7)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17, textColor=NAVY, spaceBefore=4, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, textColor=INDIGO, spaceBefore=10, spaceAfter=5)
CAP = ParagraphStyle("Cap", parent=BODY, fontName="Helvetica-Oblique", fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=10)
SMALL = ParagraphStyle("Small", parent=BODY, fontName="Helvetica", fontSize=8.3, leading=10.5, alignment=0, spaceAfter=0)
CELL = ParagraphStyle("Cell", parent=SMALL)
BUL = ParagraphStyle("Bul", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=3)
NOTE = ParagraphStyle("Note", parent=BODY, fontName="Helvetica", fontSize=9.5, leading=13, backColor=colors.HexColor("#FFF6EC"),
                      borderColor=colors.HexColor("#E67E22"), borderWidth=0.8, borderPadding=7, spaceBefore=6, spaceAfter=12)


def P(t, s=BODY): return Paragraph(t, s)
def bullets(items): return [Paragraph(i, BUL, bulletText="•") for i in items]


def table(data, widths, header=True, font=8.3, zebra=True):
    rows = [[Paragraph(str(c), CELL) if not hasattr(c, "wrap") else c for c in r] for r in data]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9CFE0")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
        for j in range(len(data[0])):
            rows[0][j] = Paragraph(f'<font color="white"><b>{data[0][j]}</b></font>', CELL)
        t = Table(rows, colWidths=widths, repeatRows=1)
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F4F5FB")))
    t.setStyle(TableStyle(st))
    return t


def fit_image(path, max_w=16.2 * cm, max_h=21 * cm):
    w, h = Image.open(path).size
    r = min(max_w / w, max_h / h)
    return RLImage(str(path), width=w * r, height=h * r)


def crop_screenshot(name):
    """Remove the empty space below the content of a real screenshot (no pixels are altered)."""
    src = SHOTS / f"{name}.png"
    im = Image.open(src).convert("RGB")
    w, h = im.size
    bg = im.getpixel((w - 5, h - 5))
    last = 0
    px = im.load()
    for y in range(h - 1, 0, -1):
        row_has = any(px[x, y] != bg for x in range(310, w - 10, 6))
        if row_has:
            last = y
            break
    im.crop((0, 0, w, min(h, last + 30))).save(TMP / f"{name}.png")
    return TMP / f"{name}.png"


def figure(name, caption, max_h=19 * cm):
    return KeepTogether([fit_image(crop_screenshot(name), max_h=max_h), P(caption, CAP)])


# ------------------------------------------------------------------ diagrams (matplotlib)
def box(ax, x, y, w, h, text, fc="#EEF0FD", ec="#5B5BD6", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color="#1F2A44")


def arrow(ax, p1, p2, text=None):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#1E2A52"))
    if text:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.12, text, ha="center", fontsize=7.5, color="#444")


def architecture_diagram():
    fig, ax = plt.subplots(figsize=(9, 6.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 8.4); ax.axis("off")
    box(ax, 4, 7.4, 4, 0.8, "User  (browser)", "#FFF6EC", "#E67E22", 10, True)
    box(ax, 3, 6.0, 6, 0.95, "Streamlit UI  (app.py)\nDashboard · Input · U1–U5 pages · Editable table", "#E8ECF8", "#1E2A52", 9, True)
    arrow(ax, (6, 7.4), (6, 6.95))
    box(ax, 0.2, 4.2, 3.4, 1.1, "llm_client.py  (U2)\nGemini / Groq free API", "#EEF0FD")
    box(ax, 0.2, 2.75, 3.4, 1.0, "fallback.py\nLabelled local templates", "#FFF6EC", "#E67E22")
    box(ax, 4.3, 4.2, 3.4, 1.1, "Risk register  (pandas)\nrisk_scoring.py · lifecycle.py", "#E9F7F0", "#2E9E6B")
    box(ax, 8.4, 4.2, 3.4, 1.1, "nist_retriever.py  (U3)\nSentence-BERT MiniLM-L6-v2", "#EEF0FD")
    box(ax, 8.4, 2.75, 3.4, 1.0, "NIST AI RMF knowledge base\n(JSON, 65 entries) + embeddings", "#F4F5FD")
    box(ax, 4.3, 2.75, 3.4, 1.0, "rmf_classifier.py  (U5)\nGOVERN·MAP·MEASURE·MANAGE", "#EEF0FD")
    box(ax, 4.3, 1.1, 3.4, 1.0, "charts.py (U4) · export.py\nHeatmap · stats · CSV", "#E9F7F0", "#2E9E6B")
    box(ax, 0.2, 1.1, 3.4, 1.0, "samples.py  (Project A, B)\nCurated registers", "#F4F5FD")
    box(ax, 8.4, 1.1, 3.4, 1.0, "Browser download\nrisk register CSV", "#FFF6EC", "#E67E22")
    arrow(ax, (5, 6.0), (2.0, 5.3)); arrow(ax, (2.0, 4.2), (2.0, 3.75), "on failure / no key")
    arrow(ax, (1.2, 4.2), (4.3, 4.7)); arrow(ax, (6, 6.0), (6, 5.3)); arrow(ax, (7.0, 6.0), (10.0, 5.3))
    arrow(ax, (10.1, 4.2), (10.1, 3.75)); arrow(ax, (7.7, 4.5), (8.4, 4.7), "")
    arrow(ax, (6, 4.2), (6, 3.75)); arrow(ax, (6, 2.75), (6, 2.1)); arrow(ax, (7.7, 1.6), (8.4, 1.6))
    arrow(ax, (2.0, 2.1), (4.3, 4.3)) if False else None
    fig.tight_layout(); p = TMP / "architecture.png"; fig.savefig(p, dpi=170); plt.close(fig); return p


def workflow_diagram():
    fig, ax = plt.subplots(figsize=(9, 2.9)); ax.set_xlim(0, 15); ax.set_ylim(0, 4.2); ax.axis("off")
    steps = [("1. Describe\nAI project", "#FFF6EC"), ("2. Generate risks\nLLM or labelled\nfallback (U2)", "#EEF0FD"),
             ("3. Map to 8\nlifecycle\nstages (U1)", "#E9F7F0"), ("4. Retrieve NIST\nguidance with\nSBERT (U3)", "#EEF0FD"),
             ("5. Score, heatmap\nand export CSV\n(U4)", "#E9F7F0"), ("6. Classify into\nGOVERN/MAP/\nMEASURE/MANAGE (U5)", "#EEF0FD")]
    for i, (t, c) in enumerate(steps):
        x = 0.1 + i * 2.45
        box(ax, x, 1.2, 2.1, 1.8, t, c, "#5B5BD6", 8.3)
        if i < len(steps) - 1:
            arrow(ax, (x + 2.1, 2.1), (x + 2.45, 2.1))
    ax.text(7.5, 0.55, "User can edit the register at any time; scores, severity and RMF classification recalculate automatically.",
            ha="center", fontsize=8, style="italic", color="#444")
    fig.tight_layout(); p = TMP / "workflow.png"; fig.savefig(p, dpi=170); plt.close(fig); return p


def heatmap_png(key, title):
    import numpy as np
    grid = np.array(RES["samples"][key]["heatmap"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for L in range(1, 6):
        for I in range(1, 6):
            s = L * I
            c = SEVERITY_COLORS["Low" if s <= 4 else "Moderate" if s <= 9 else "High" if s <= 16 else "Critical"]
            ax.add_patch(plt.Rectangle((I - 0.5, L - 0.5), 1, 1, fc=c, ec="white", lw=2))
            n = grid[L - 1, I - 1]
            if n:
                ax.text(I, L, str(n), ha="center", va="center", color="white", fontsize=16, fontweight="bold")
    ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
    ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6)); ax.set_xlabel("Impact"); ax.set_ylabel("Likelihood")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); p = TMP / f"heat_{key}.png"; fig.savefig(p, dpi=170); plt.close(fig); return p


def rmf_png(key):
    c = RES["samples"][key]["rmf_counts"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    fs = list(c.keys()); ax.bar(fs, [c[f] for f in fs], color=[RMF_COLORS[f] for f in fs])
    for i, f in enumerate(fs):
        ax.text(i, c[f] + 0.1, str(c[f]), ha="center", fontsize=10)
    ax.set_ylabel("Risks"); ax.set_title("Risks by RMF function", fontsize=10)
    fig.tight_layout(); p = TMP / f"rmf_{key}.png"; fig.savefig(p, dpi=170); plt.close(fig); return p


# ------------------------------------------------------------------ page furniture
def on_page(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#6B7794"))
    canvas.drawString(2.2 * cm, 1.3 * cm, f"{STUDENT['name']} · Roll {STUDENT['roll']} · P55 · NIST AI RMF Risk Register Builder")
    canvas.drawRightString(A4[0] - 2.2 * cm, 1.3 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#C9CFE0")); canvas.line(2.2 * cm, 1.7 * cm, A4[0] - 2.2 * cm, 1.7 * cm)
    canvas.restoreState()


def sample_section(key, letter, extra_text):
    s = RES["samples"][key]; st = s["stats"]
    f = [P(f"Project {letter}: {s['name']}", H2), P(extra_text)]
    bs = st["by_severity"]
    f.append(P(f"The register contains <b>{st['total']}</b> risks covering all eight lifecycle stages. Severity distribution: "
               f"<b>{bs['Critical']}</b> Critical, <b>{bs['High']}</b> High, <b>{bs['Moderate']}</b> Moderate, <b>{bs['Low']}</b> Low. "
               f"Average risk score is <b>{st['average_score']}</b> and the maximum is <b>{st['max_score']}</b> out of 25. "
               "The risks in this register were written by hand by the project author (labelled \"Curated sample\" in the application); "
               "they were <b>not</b> generated by an LLM."))
    rows = [["ID", "Title", "Lifecycle stage", "L", "I", "Score", "Severity"]]
    for r in s["register"]:
        rows.append([r["Risk ID"], r["Title"], r["Lifecycle Stage"], r["Likelihood"], r["Impact"], r["Risk Score"], r["Severity"]])
    f.append(table(rows, [1.0 * cm, 7.3 * cm, 3.0 * cm, 0.7 * cm, 0.7 * cm, 1.2 * cm, 1.9 * cm]))
    f.append(P(f"Table: Risk register of Project {letter} (L = likelihood, I = impact; values from the application's own data).", CAP))
    h = RLImage(str(heatmap_png(key, f"Project {letter} heatmap")), width=8.3 * cm, height=7.0 * cm)
    r_ = RLImage(str(rmf_png(key)), width=7.6 * cm, height=5.3 * cm)
    f.append(Table([[h, r_]], colWidths=[8.6 * cm, 8.0 * cm]))
    f.append(P(f"Figure: Project {letter} heatmap (cell colour = severity band, number = count of risks) and RMF function distribution "
               "(GOVERN {GOVERN}, MAP {MAP}, MEASURE {MEASURE}, MANAGE {MANAGE}). Charts redrawn with matplotlib from the same register data "
               "that the application displays.".format(**s["rmf_counts"]), CAP))
    if s["retrieval"]:
        top = [r for r in s["retrieval"] if r["Rank"] == 1]
        rows = [["Risk", "Top NIST guidance (Sentence-BERT)", "RMF function", "Cosine similarity"]]
        titles = {r["Risk ID"]: r["Title"] for r in s["register"]}
        for r in top:
            rows.append([f"{r['Risk ID']} {titles[r['Risk ID']][:42]}", f"{r['Guidance ID']}: {r['Guidance Title']}", r["RMF Function"], f"{r['Similarity']:.3f}"])
        f.append(table(rows, [5.6 * cm, 6.2 * cm, 2.3 * cm, 2.6 * cm]))
        f.append(P("Table: Top-1 retrieved NIST guidance for each risk (real Sentence-BERT output).", CAP))
    else:
        f.append(P("<b>Sentence-BERT retrieval results:</b> not included. The model could not be downloaded in the environment where this "
                   "report was generated, so no retrieval was run and no similarity scores are reported.", NOTE))
    return f


# ================================================================== CONTENT
def build():
    story = []
    # ---------------- cover
    story += [Spacer(1, 1.2 * cm), P("MIT Art, Design and Technology University", ParagraphStyle("c0", parent=H1, alignment=TA_CENTER, fontSize=15)),
              P("School of Computing · Generative AI · Individual CA Mini Project", ParagraphStyle("c1", parent=BODY, alignment=TA_CENTER, fontName="Helvetica")),
              Spacer(1, 1.6 * cm),
              P("NIST AI RMF Risk Register Builder", ParagraphStyle("c2", parent=H1, alignment=TA_CENTER, fontSize=28, leading=34, textColor=INDIGO)),
              P("An LLM and Sentence-BERT based web application that turns an AI project description into a structured risk register aligned with the NIST AI Risk Management Framework",
                ParagraphStyle("c3", parent=BODY, alignment=TA_CENTER, fontName="Helvetica-Oblique", fontSize=11.5, leading=16)),
              Spacer(1, 1.6 * cm)]
    info = [["Student name", STUDENT["name"]], ["Roll number", STUDENT["roll"]], ["Enrollment number", STUDENT["enrollment"]],
            ["University", STUDENT["university"]], ["Subject", STUDENT["subject"]], ["Problem statement", STUDENT["problem"]],
            ["Project title", "NIST AI RMF Risk Register Builder"], ["Type of work", "Individual CA Mini Project"]]
    t = Table([[Paragraph(f"<b>{a}</b>", ParagraphStyle("k", parent=BODY, fontName="Helvetica", fontSize=10.5)),
                Paragraph(b, ParagraphStyle("v", parent=BODY, fontName="Helvetica", fontSize=10.5))] for a, b in info], colWidths=[5.2 * cm, 10.2 * cm])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9CFE0")), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F4F5FB")),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story += [t, Spacer(1, 1.5 * cm), P(f"Report generated on {RES['generated_at'][:10]}", ParagraphStyle("c4", parent=BODY, alignment=TA_CENTER, fontName="Helvetica", fontSize=9)), PageBreak()]

    # ---------------- contents
    story.append(P("Contents", H1))
    toc = ["1. Abstract", "2. Introduction", "3. Problem Statement and Objectives", "4. Existing System and Limitations", "5. Proposed System",
           "6. Technology Stack", "7. System Architecture and Workflow", "8. Detailed Implementation of U1–U5", "9. NIST AI RMF Explained",
           "10. Sentence-BERT Methodology", "11. Risk Scoring and Heatmap", "12. Healthcare Sample Project Results",
           "13. Recruitment Sample Project Results", "14. Application Screenshots", "15. Testing and Results", "16. Limitations and Future Scope",
           "17. Conclusion", "18. References"]
    story += [P(x, ParagraphStyle("t", parent=BODY, fontName="Helvetica", fontSize=11, spaceAfter=6, alignment=0)) for x in toc]
    story.append(P("<b>Honesty note.</b> This report only describes features that exist in the submitted application and results that were actually "
                   "produced. Where something could not be run in the build environment (see Sections 10, 14 and 15), the report states this explicitly.", NOTE))
    story.append(PageBreak())

    # ---------------- 1 abstract
    story += [P("1. Abstract", H1),
              P("Organisations that build or buy artificial-intelligence systems are increasingly expected to identify, assess and manage the risks those systems create. "
                "The NIST AI Risk Management Framework (AI RMF 1.0) provides a voluntary, widely referenced structure for this work, but turning it into a project-specific "
                "risk register is slow, manual and depends on expert knowledge. This project presents <b>NIST AI RMF Risk Register Builder</b>, a Streamlit web application "
                "that accepts a plain-language description of an AI project and produces a structured risk register."),
              P("The application implements five tasks. <b>U1</b> maps risks to eight AI lifecycle stages. <b>U2</b> uses a large language model (Google Gemini or Groq free-tier APIs) "
                "to generate project-specific risks with a likelihood, impact, score, severity, mitigation and responsible stakeholder; when no API key is available a clearly "
                "labelled template-based fallback is used instead. <b>U3</b> embeds each risk with the Sentence-BERT model <i>all-MiniLM-L6-v2</i> and retrieves the most similar guidance from a "
                f"local knowledge base of {RES['knowledge_base_entries']} entries paraphrasing the NIST AI RMF. <b>U4</b> calculates risk score as Likelihood × Impact, classifies severity, draws an interactive 5×5 heatmap and exports "
                "the complete register as CSV. <b>U5</b> classifies each risk under the GOVERN, MAP, MEASURE and MANAGE functions with a written rationale, keeping lifecycle stages separate from RMF functions."),
              P("Two complete sample projects, an AI-based healthcare diagnosis system and an AI-based recruitment screening system, each with a 16-risk register, demonstrate the workflow. "
                f"The automated test suite reports <b>{RES['tests']['summary']}</b>. "
                + ("Sentence-BERT retrieval was executed and its results are reported." if SBERT_RAN else
                   "In the environment used to prepare this report the Sentence-BERT model could not be downloaded, so the U3 retrieval code was verified with a test double and "
                   "its real-model tests were skipped; the report does not present any retrieval scores.")),
              PageBreak()]

    # ---------------- 2 intro
    story += [P("2. Introduction", H1),
              P("AI systems now influence decisions about health, employment, credit and safety. When such systems fail, the harm can be large and can fall unevenly on different groups. "
                "Risk management for AI therefore differs from ordinary software risk management: models learn from data, behave probabilistically, change as the world changes and are often hard to explain."),
              P("In January 2023 the U.S. National Institute of Standards and Technology (NIST) published the AI Risk Management Framework (AI 100-1). It is voluntary and sector-neutral. "
                "It describes characteristics of trustworthy AI and organises risk-management activities into four core functions: GOVERN, MAP, MEASURE and MANAGE. In July 2024 NIST added a Generative AI Profile (AI 600-1)."),
              P("A <b>risk register</b> is the standard tool for recording risks: what could go wrong, how likely and how severe it is, who owns it and what will be done. This project automates the creation of such a register for AI projects and links each risk to the NIST guidance that helps address it."),
              P("Scope and honesty. The knowledge base is a student paraphrase of NIST subcategories, not a reproduction of the official text, and the tool supports human judgement rather than replacing a formal risk assessment."),
              PageBreak()]

    # ---------------- 3 problem
    story += [P("3. Problem Statement and Objectives", H1),
              P("<b>Problem statement (P55).</b> Build a web application that accepts an AI project description and automatically creates a structured AI risk register using the NIST AI Risk Management Framework."),
              P("<b>Objectives</b>", H2)]
    story += bullets(["Identify AI lifecycle stages and map potential risks to each stage (U1).",
                      "Generate relevant, project-specific risks with an LLM, each with ID, title, description, stage, category, likelihood, impact, score, severity, mitigation and stakeholder (U2).",
                      "Retrieve matching NIST guidance for each risk using genuine Sentence-BERT semantic search over a local knowledge base, showing similarity scores, RMF functions and references (U3).",
                      "Calculate risk score as Likelihood × Impact, show a 5×5 interactive heatmap and statistics, and export the full register as CSV (U4).",
                      "Organise risks under GOVERN, MAP, MEASURE and MANAGE with categories, subcategories and an explanation, separate from lifecycle stages (U5).",
                      "Stay demonstrable without an API key through a clearly labelled fallback, and never present fallback output as LLM output.",
                      "Provide two complete worked examples and a deployable, documented project."])
    story.append(PageBreak())

    # ---------------- 4 existing
    story += [P("4. Existing System and Limitations", H1),
              P("Today AI risk registers are usually produced in one of three ways:"),
              ]
    story += bullets(["<b>Spreadsheets and templates</b> filled in by hand in workshops. They are flexible but slow, inconsistent between teams and rely on the experience of the people in the room.",
                      "<b>Generic enterprise GRC tools</b> that track risks well but contain no AI-specific knowledge; the user must already know which AI risks to enter.",
                      "<b>Reading the NIST documents directly</b>. The AI RMF and its Playbook are detailed, and linking a specific project risk to the right subcategory takes time."])
    story += [P("<b>Limitations of these approaches</b>", H2)]
    story += bullets(["Risks are generic rather than specific to the project's data, users and deployment.",
                      "No consistent link between a risk and the framework guidance that addresses it.",
                      "Lifecycle stage, severity and framework function are often mixed together, making the register hard to analyse.",
                      "No instant visual overview (heatmap, statistics) and manual effort to keep scores consistent after edits."])
    story += [PageBreak()]

    # ---------------- 5 proposed
    story += [P("5. Proposed System", H1),
              P("The proposed system is a single web application with a guided workflow. The user describes an AI project (or loads one of two samples). The system generates a register, "
                "maps every risk to a lifecycle stage, retrieves NIST guidance by semantic similarity, scores and visualises the risks, classifies them under the four RMF functions and lets the user edit and export the result."),
              P("<b>Key design decisions</b>", H2)]
    story += bullets(["<b>Modular code.</b> One Python module per task (lifecycle, LLM client, fallback, retriever, scoring, classifier, charts, export), so each can be explained and tested separately.",
                      "<b>Honest labelling.</b> Every risk has a 'Generated By' value: <i>LLM – Gemini/Groq</i>, <i>Local template fallback (not LLM)</i>, <i>Curated sample (hand-authored)</i> or <i>Added manually by user</i>.",
                      "<b>No silent substitution.</b> If Sentence-BERT cannot be loaded, the U3 page says so; it does not fall back to keyword matching.",
                      "<b>Scores cannot drift.</b> Score and severity are recomputed from Likelihood and Impact after generation and after every edit.",
                      "<b>Secrets stay out of code.</b> API keys are read only from environment variables or Streamlit secrets."])
    story.append(PageBreak())

    # ---------------- 6 tech
    story += [P("6. Technology Stack", H1),
              table([["Layer", "Technology", "Purpose"],
                     ["Language", "Python 3.10+", "All application logic"],
                     ["Web UI", "Streamlit", "Dashboard, forms, navigation, editable table (st.data_editor), CSV download"],
                     ["Data", "Pandas, NumPy", "Risk register table, statistics, matrix and vector maths"],
                     ["Embeddings", "sentence-transformers, model all-MiniLM-L6-v2", "384-dimensional Sentence-BERT sentence embeddings for U3"],
                     ["Charts", "Plotly", "Interactive 5×5 heatmap, donut, bar charts"],
                     ["LLM (free tier)", "Google Gemini API or Groq API via requests", "U2 project-specific risk generation (JSON output)"],
                     ["Fallback", "Local template library", "Labelled non-LLM generation when no key / API failure"],
                     ["Knowledge base", "JSON file", "65 paraphrased NIST AI RMF entries with IDs, functions and references"],
                     ["Testing", "pytest, Streamlit AppTest", "Unit and end-to-end UI tests"],
                     ["Deployment", "Streamlit Community Cloud + GitHub", "Free hosting (guide provided; deployment not performed for this report)"],
                     ["Report tooling", "ReportLab, Matplotlib, Playwright", "PDF generation and real screenshot capture"]],
                    [3.0 * cm, 5.4 * cm, 8.2 * cm]),
              PageBreak()]

    # ---------------- 7 architecture
    story += [P("7. System Architecture and Workflow", H1),
              P("The application follows a layered design. The Streamlit front end calls independent modules; the central data structure is a pandas DataFrame (the risk register) that every module reads or extends."),
              fit_image(architecture_diagram(), max_h=12.5 * cm), P("Figure: System architecture of the NIST AI RMF Risk Register Builder.", CAP),
              fit_image(workflow_diagram(), max_h=6 * cm), P("Figure: End-to-end workflow from project description to exported register.", CAP),
              PageBreak()]

    # ---------------- 8 U1-U5
    story += [P("8. Detailed Implementation of U1–U5", H1)]
    story += [P("U1 – Lifecycle Risk Mapping", H2),
              P("<i>Module: modules/lifecycle.py.</i> Eight stages are defined in config.py: Planning, Data Collection, Data Preparation, Development, Evaluation, Deployment, Monitoring and Retirement. "
                "Each risk carries exactly one stage. Free text returned by an LLM (for example 'data prep') is normalised to one of the eight stages. The U1 page shows a stage summary table, a stacked bar of severity per stage, "
                "expandable lists per stage and a coverage check for empty stages.")]
    story += [P("U2 – LLM-Based Risk Generation", H2),
              P("<i>Modules: llm_client.py, fallback.py.</i> A structured prompt includes the project name, description, domain, data, users and deployment context and asks for a JSON array of risks with the required fields. "
                "The reply is parsed tolerantly (code fences and extra text are removed), every field is validated (stage normalised, category checked against nine allowed categories, likelihood and impact clamped to 1–5) and score and severity are computed in code, not trusted from the model. "
                "Gemini or Groq is chosen according to which key (GEMINI_API_KEY or GROQ_API_KEY) is found in environment variables or Streamlit secrets. If there is no key, or the call fails, the user sees the error and the fallback produces the register, labelled "
                "'Local template fallback (not LLM)'. The fallback picks pre-written templates using simple domain detection and guarantees at least one risk per lifecycle stage."),
              P("Each risk row contains: Risk ID, Title, Description, Lifecycle Stage, Risk Category, Likelihood, Impact, Risk Score, Severity, Mitigation Strategy, Responsible Stakeholder and Generated By.")]
    story += [P("U3 – Sentence-BERT Semantic Retrieval", H2),
              P("<i>Module: modules/nist_retriever.py.</i> The knowledge base file data/nist_ai_rmf_knowledge_base.json holds guidance entries with ID, RMF function, category, title, paraphrased text and reference. "
                "Entries and risks are encoded by <i>sentence-transformers/all-MiniLM-L6-v2</i> into normalised 384-dimensional vectors; similarity is the dot product (equal to cosine similarity). The top-k entries (default 3) are returned with similarity, function and reference. "
                "Knowledge-base embeddings are cached on disk. No keyword matching is used anywhere in retrieval; if the model cannot load, a clear error is displayed.")]
    story += [P("U4 – Risk Heatmap and CSV Export", H2),
              P("<i>Modules: risk_scoring.py, charts.py, export.py.</i> Score = Likelihood × Impact. Severity bands: 1–4 Low, 5–9 Moderate, 10–16 High, 17–25 Critical. The 5×5 Plotly heatmap colours each cell by the severity band of its score, prints the number of risks in the cell and shows risk IDs and titles on hover. "
                "Dashboard cards show totals, counts per severity, average and maximum score. The full register (including RMF classification) downloads as UTF-8 CSV with a byte-order mark so it opens correctly in Excel.")]
    story += [P("U5 – NIST AI RMF Classification", H2),
              P("<i>Module: modules/rmf_classifier.py.</i> A transparent rule table maps each risk category (and, for some stage–category pairs, the lifecycle stage) to a primary RMF function and subcategory, with a written rationale shown to the user. "
                "High and Critical risks also get a MANAGE treatment note. When U3 has run, the best-matching guidance is shown as separate 'semantic evidence' and never overrides the rules. "
                "Lifecycle stage and RMF function are always separate columns."),
              PageBreak()]

    # ---------------- 9 NIST
    story += [P("9. NIST AI RMF Explained", H1),
              P("The AI RMF 1.0 (NIST AI 100-1) has two parts. Part 1 explains how to frame AI risk and describes seven characteristics of trustworthy AI: valid and reliable; safe; secure and resilient; accountable and transparent; explainable and interpretable; privacy-enhanced; and fair with harmful bias managed. "
                "Part 2 is the <b>Core</b>, made of four functions, each divided into categories and subcategories."),
              table([["Function", "Purpose", "Example subcategories used in this project"],
                     ["GOVERN", "Cross-cutting culture, policies, accountability and oversight", "GOVERN 1.1 legal requirements; GOVERN 1.7 decommissioning; GOVERN 3.2 human-AI oversight roles"],
                     ["MAP", "Establish context; identify risks and impacts", "MAP 1.1 intended purpose and context; MAP 2.3 data representativeness; MAP 4.1 third-party legal risks"],
                     ["MEASURE", "Analyse, test and monitor risks with metrics", "MEASURE 2.5 valid and reliable; 2.7 security; 2.9 explainability; 2.10 privacy; 2.11 fairness; 2.4 production monitoring"],
                     ["MANAGE", "Prioritise and treat risks; respond and recover", "MANAGE 1.2 prioritisation; 1.3 responses to high-priority risks; 2.3 unknown risks; 2.4 deactivate; 4.3 incidents"]],
                    [2.2 * cm, 5.2 * cm, 9.2 * cm]),
              Spacer(1, 6),
              P("<b>Lifecycle vs. function.</b> The AI RMF describes an AI lifecycle (plan and design, collect and process data, build and use the model, verify and validate, deploy and use, operate and monitor, use or impacted by) "
                "and says the four functions apply throughout it. This project therefore treats <i>when</i> a risk arises (lifecycle stage, U1) and <i>what kind of risk-management activity</i> addresses it (RMF function, U5) as two separate dimensions."),
              P("The project's eight lifecycle stages are a simplified teaching view chosen to match the assignment; they are not the exact stage names of the NIST figure."),
              PageBreak()]

    # ---------------- 10 SBERT
    story += [P("10. Sentence-BERT Methodology", H1),
              P("Sentence-BERT (Reimers and Gurevych, 2019) adapts the BERT transformer with a siamese network so that whole sentences are mapped to fixed-size vectors whose distance reflects meaning. "
                "The model <i>all-MiniLM-L6-v2</i> is a small, fast 6-layer model that produces 384-dimensional embeddings and is trained on over a billion sentence pairs."),
              P("<b>Method used</b>", H2)]
    story += bullets(["Build the text for each knowledge-base entry as 'title. text' and for each risk as 'title. description'.",
                      "Encode all texts with the model and L2-normalise the vectors.",
                      "Compute similarity = query vector · knowledge-base matrix (cosine similarity because vectors are unit length).",
                      "Sort by similarity and return the top-k entries with ID, RMF function, category, reference and score.",
                      "Cache the knowledge-base embeddings (hash of text + model name) to avoid recomputation."])
    story += [P("<b>Why semantic search rather than keywords?</b> A risk such as 'the model treats women and minority groups worse' shares few words with the guidance 'fairness and bias are evaluated across groups', "
                "but their meanings are close. Embedding similarity can connect them; keyword matching cannot.")]
    if SBERT_RAN:
        story += [P("<b>Execution status:</b> the model was downloaded and retrieval was executed. Results appear in Sections 12 and 13.")]
    else:
        story += [P(f"<b>Execution status – please read.</b> The model download was blocked in the environment where this report was prepared ({RES['sbert']['error']}). "
                    "The retrieval code path is complete and unit-tested with a deterministic test double that checks only the maths (normalisation, ranking, output columns). "
                    "Three tests that load the real model were skipped automatically. <b>No similarity scores from the real model are reported in this document.</b> "
                    "Running the application or tools/collect_results.py on a computer with internet access executes the real model; re-running tools/build_report.py then fills these sections with real results.", NOTE)]
    story.append(PageBreak())

    # ---------------- 11 scoring
    story += [P("11. Risk Scoring and Heatmap", H1),
              P("<b>Scales.</b> Likelihood: 1 Rare, 2 Unlikely, 3 Possible, 4 Likely, 5 Almost certain. Impact: 1 Negligible, 2 Minor, 3 Moderate, 4 Major, 5 Severe."),
              P("<b>Formula.</b> Risk Score = Likelihood × Impact (range 1–25)."),
              table([["Score range", "Severity", "Typical response"], ["1 – 4", "Low", "Accept and monitor"], ["5 – 9", "Moderate", "Plan mitigation in normal work"],
                     ["10 – 16", "High", "Mitigation required with named owner (MANAGE 1.2 / 1.3)"], ["17 – 25", "Critical", "Immediate action; do not deploy until treated"]],
                    [3.0 * cm, 3.0 * cm, 10.6 * cm]),
              Spacer(1, 6),
              P("The heatmap places each risk in the cell of its likelihood (rows) and impact (columns). Because severity depends only on the product, the colour layout is fixed (green, yellow, orange, red) and the numbers show how many risks fall in each cell. "
                "In the application the heatmap is interactive: hovering over a cell lists the risk IDs and titles in it. Editing likelihood or impact in the register recalculates score, severity, heatmap and statistics instantly."),
              P("The scoring logic is verified by tests that check every one of the 25 likelihood–impact combinations, the band boundaries (4/5, 9/10, 16/17) and rejection of out-of-range scores."),
              PageBreak()]

    # ---------------- 12 healthcare
    story += [P("12. Healthcare Sample Project Results", H1)]
    story += sample_section("healthcare_diagnosis", "A",
                            "Project A is an AI decision-support tool that analyses patient records, lab results and chest X-ray images to suggest probable diagnoses to doctors. "
                            "Its register covers dataset bias, incorrect predictions, patient privacy, explainability, model drift, cybersecurity and human oversight, together with planning, labelling and retirement risks.")
    story.append(PageBreak())

    # ---------------- 13 recruitment
    story += [P("13. Recruitment Sample Project Results", H1)]
    story += sample_section("recruitment_screening", "B",
                            "Project B is an AI system that parses resumes, scores and ranks candidates and automatically rejects those below a threshold. "
                            "Its register covers demographic bias, unfair candidate ranking, privacy violations, incorrect rejection, transparency, security and human oversight, plus planning, data-quality and retirement risks.")
    story.append(PageBreak())

    # ---------------- 14 screenshots
    story += [P("14. Application Screenshots", H1),
              P("All images below are real screenshots of the running application, captured with headless Chrome through tools/capture_screenshots.py. Empty space below each page has been cropped; no content has been edited.")]
    story += [figure("01_dashboard_empty", "Figure: Dashboard before any project is loaded.", 8 * cm),
              figure("02_project_input", "Figure: AI project input form with generation mode selector.", 12 * cm), PageBreak(),
              figure("healthcare_01_dashboard", "Figure: Dashboard for Project A (statistics, charts, top risks).", 20 * cm), PageBreak(),
              figure("healthcare_02_lifecycle_U1", "Figure: U1 lifecycle mapping for Project A.", 20 * cm), PageBreak(),
              figure("healthcare_03_register_U2", "Figure: U2 editable risk register with filters and CSV download.", 20 * cm), PageBreak(),
              figure("healthcare_04_nist_guidance_U3",
                     "Figure: U3 page as captured in the build environment. " + ("" if SBERT_RAN else
                     "The Sentence-BERT model could not be downloaded there, so the page honestly reports that retrieval has not run. A screenshot with real results must be captured on a computer with internet access."), 9 * cm),
              figure("healthcare_05_heatmap_U4", "Figure: U4 interactive heatmap, severity cards and ranked table.", 20 * cm), PageBreak(),
              figure("healthcare_06_rmf_U5", "Figure: U5 RMF classification for Project A.", 20 * cm), PageBreak(),
              figure("recruitment_01_dashboard", "Figure: Dashboard for Project B.", 20 * cm), PageBreak(),
              figure("recruitment_05_heatmap_U4", "Figure: U4 heatmap and statistics for Project B.", 20 * cm), PageBreak(),
              figure("recruitment_06_rmf_U5", "Figure: U5 RMF classification for Project B.", 20 * cm), PageBreak()]

    # ---------------- 15 testing
    cases = RES["tests"]["cases"]
    by_file = {}
    for c in cases:
        d = by_file.setdefault(c["file"], {"PASSED": 0, "FAILED": 0, "SKIPPED": 0}); d[c["status"]] += 1
    desc = {"test_scoring_and_lifecycle.py": "U1 stage mapping, U4 scoring bands, all 25 L×I combinations, heatmap counts, statistics",
            "test_llm_fallback_samples.py": "U2 JSON parsing and validation, no-key behaviour, mocked LLM labelled as LLM, fallback labelled 'not LLM', sample structure and topic coverage",
            "test_retrieval_rmf_export.py": "Knowledge-base integrity, U3 retrieval maths (test double), real-model tests, U5 classification, CSV round trip",
            "test_app_ui.py": "Headless end-to-end run of every page for both samples, fallback generation, empty-input validation"}
    rows = [["Test file", "What it checks", "Passed", "Failed", "Skipped"]]
    for f, d in by_file.items():
        rows.append([f, desc.get(f, ""), d["PASSED"], d["FAILED"], d["SKIPPED"]])
    story += [P("15. Testing and Results", H1),
              P(f"The automated suite was executed with pytest on {RES['generated_at'][:10]}. Result: <b>{RES['tests']['summary']}</b> in {RES['tests']['seconds']} s. The figures below are parsed from that run."),
              table(rows, [4.4 * cm, 7.6 * cm, 1.5 * cm, 1.5 * cm, 1.6 * cm]), Spacer(1, 6)]
    if RES["tests"]["skip_reasons"]:
        story += [P("<b>Skipped tests.</b> The following tests were skipped automatically (not faked) because the real Sentence-BERT model could not be downloaded in the test environment:"),
                  ] + bullets([f"{s.split(':')[0].replace('SKIPPED [1] ', '')} – model unavailable" for s in RES["tests"]["skip_reasons"]])
    story += [P("<b>Manual and UI verification.</b> Every page was opened in headless Chrome for both sample projects and screenshots were taken. During this process a legibility defect was found "
                "(dark text on the dark sidebar for the sample dropdown and warning box) and fixed before the final screenshots were captured."),
              P("<b>What was not tested.</b> A live call to Gemini or Groq was not made for this report (it requires a private API key); the LLM path is tested with a mocked response, which verifies parsing, validation and labelling but not the real model's output quality. "
                "Deployment to Streamlit Community Cloud was not performed, so no live URL is given."),
              PageBreak()]

    # ---------------- 16 limits
    story += [P("16. Limitations and Future Scope", H1), P("<b>Limitations</b>", H2)]
    story += bullets(["LLM output quality depends on the model and on how detailed the project description is; risks must be reviewed by a human before use.",
                      "The knowledge base paraphrases selected NIST subcategories (65 entries) rather than the complete Playbook; wording should be checked against the official documents.",
                      "The RMF classification is rule-based and heuristic; the semantic match is shown as supporting evidence only.",
                      "Likelihood and impact are qualitative 1–5 judgements, not calibrated probabilities.",
                      "The sample registers are hand-authored, not LLM-generated, and the sample datasets in sample_projects/ are synthetic.",
                      "The Sentence-BERT model needs about 90 MB of download on first use and a few hundred MB of memory; the free Streamlit Cloud tier may be slow to start.",
                      "Single-user, session-based: registers are not saved on the server between sessions (use CSV export)."])
    story += [P("<b>Future scope</b>", H2)]
    story += bullets(["Import the complete NIST Playbook and the Generative AI Profile (AI 600-1) into the knowledge base.",
                      "Add retrieval-augmented generation so the LLM drafts mitigations grounded in the retrieved guidance.",
                      "Add Word/PDF export of the register, risk owners' sign-off and version history.",
                      "Compare a re-ranker or larger embedding model, and evaluate retrieval quality with expert-labelled pairs.",
                      "Residual-risk scoring after mitigation and tracking of mitigation status.",
                      "Support other frameworks (ISO/IEC 42001, EU AI Act risk classes) alongside the NIST AI RMF."])
    story.append(PageBreak())

    # ---------------- 17 conclusion
    story += [P("17. Conclusion", H1),
              P("The project delivers a working, modular web application that converts an AI project description into a structured NIST AI RMF risk register. "
                "All five tasks are implemented: lifecycle mapping (U1), LLM generation with a clearly labelled fallback (U2), Sentence-BERT retrieval over a local NIST knowledge base (U3), "
                "scoring with an interactive heatmap and CSV export (U4) and GOVERN/MAP/MEASURE/MANAGE classification (U5). Two complete sample projects demonstrate the full workflow."),
              P(f"The automated tests ({RES['tests']['summary']}) and the headless UI run confirm that the application works as designed. "
                + ("" if SBERT_RAN else "The one limitation of this build is that the real Sentence-BERT model and a live LLM call could not be exercised in the environment where the report was prepared; "
                   "both are exercised simply by running the application on an internet-connected computer with an API key, and the report can then be regenerated with real results.")),
              P("Beyond the technical work, the project shows that a framework such as the NIST AI RMF becomes far easier to apply when risks are generated from the project's own context and linked automatically to concrete guidance."),
              PageBreak()]

    # ---------------- 18 references
    story += [P("18. References", H1)]
    refs = ["National Institute of Standards and Technology (2023). <i>Artificial Intelligence Risk Management Framework (AI RMF 1.0)</i>, NIST AI 100-1. https://doi.org/10.6028/NIST.AI.100-1",
            "National Institute of Standards and Technology (2024). <i>Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile</i>, NIST AI 600-1. https://doi.org/10.6028/NIST.AI.600-1",
            "NIST AI Resource Center. <i>NIST AI RMF Playbook</i>. https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook",
            "Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. <i>Proceedings of EMNLP-IJCNLP 2019</i>. https://arxiv.org/abs/1908.10084",
            "Hugging Face. <i>sentence-transformers/all-MiniLM-L6-v2</i> model card. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
            "Streamlit Inc. <i>Streamlit documentation</i>. https://docs.streamlit.io",
            "Plotly Technologies Inc. <i>Plotly Python graphing library</i>. https://plotly.com/python/",
            "Google. <i>Gemini API documentation</i>. https://ai.google.dev/gemini-api/docs",
            "Groq. <i>GroqCloud API documentation</i>. https://console.groq.com/docs"]
    story += [P(f"[{i}] {r}", ParagraphStyle("ref", parent=BODY, alignment=0, leftIndent=18, firstLineIndent=-18, fontSize=10.5)) for i, r in enumerate(refs, 1)]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.0 * cm, bottomMargin=2.2 * cm,
                            title="NIST AI RMF Risk Register Builder - P55", author=STUDENT["name"])
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print("Saved", OUT)


if __name__ == "__main__":
    build()
