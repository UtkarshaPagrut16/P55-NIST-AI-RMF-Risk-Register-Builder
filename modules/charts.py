"""Plotly charts for the dashboard (U4 heatmap + supporting charts)."""
import pandas as pd
import plotly.graph_objects as go

from .config import (
    IMPACT_LABELS, LIFECYCLE_STAGES, LIKELIHOOD_LABELS, RMF_COLORS, RMF_FUNCTIONS,
    SEVERITY_COLORS, SEVERITY_ORDER,
)
from .risk_scoring import classify_severity

FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color="#1F2A44")


def _base_layout(fig, height=420, title=None):
    fig.update_layout(
        height=height, title=dict(text=title, x=0.01, font=dict(size=16)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=FONT,
        margin=dict(l=10, r=10, t=50 if title else 10, b=10),
    )
    return fig


def risk_heatmap(df: pd.DataFrame, height=520) -> go.Figure:
    """5x5 likelihood-vs-impact heatmap. Cell colour = severity band of L x I,
    cell label = number of risks, hover = the risk IDs and titles in that cell."""
    z, text, hover = [], [], []
    for L in range(1, 6):
        z_row, t_row, h_row = [], [], []
        for I in range(1, 6):
            score = L * I
            z_row.append(score)
            cell = df[(df["Likelihood"] == L) & (df["Impact"] == I)] if not df.empty else df
            t_row.append(str(len(cell)) if len(cell) else "")
            lines = "<br>".join(f"{r['Risk ID']}: {r['Title'][:45]}" for _, r in cell.iterrows()) or "No risks"
            h_row.append(
                f"<b>Likelihood {L} ({LIKELIHOOD_LABELS[L]})</b><br>Impact {I} ({IMPACT_LABELS[I]})"
                f"<br>Score {score} - {classify_severity(score)}<br>{lines}"
            )
        z.append(z_row); text.append(t_row); hover.append(h_row)

    # discrete colour bands that match the severity thresholds (1-4, 5-9, 10-16, 17-25)
    c = [SEVERITY_COLORS[s] for s in SEVERITY_ORDER]
    colorscale = [
        [0.00, c[0]], [4.5 / 25, c[0]],
        [4.5 / 25, c[1]], [9.5 / 25, c[1]],
        [9.5 / 25, c[2]], [16.5 / 25, c[2]],
        [16.5 / 25, c[3]], [1.00, c[3]],
    ]
    fig = go.Figure(go.Heatmap(
        z=z, x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5], zmin=0, zmax=25, colorscale=colorscale,
        text=text, texttemplate="%{text}", textfont=dict(size=22, color="white"),
        hovertext=hover, hoverinfo="text", showscale=False, xgap=3, ygap=3,
    ))
    fig.update_xaxes(title="Impact", tickvals=[1, 2, 3, 4, 5],
                     ticktext=[f"{i}<br>{IMPACT_LABELS[i]}" for i in range(1, 6)])
    fig.update_yaxes(title="Likelihood", tickvals=[1, 2, 3, 4, 5],
                     ticktext=[f"{i} {LIKELIHOOD_LABELS[i]}" for i in range(1, 6)])
    return _base_layout(fig, height, "Risk Heatmap: Likelihood x Impact")


def severity_donut(stats: dict) -> go.Figure:
    labels = SEVERITY_ORDER
    fig = go.Figure(go.Pie(
        labels=labels, values=[stats["by_severity"][s] for s in labels], hole=0.62,
        marker=dict(colors=[SEVERITY_COLORS[s] for s in labels]), sort=False,
        textinfo="value", hovertemplate="%{label}: %{value} risks<extra></extra>",
    ))
    fig.update_layout(legend=dict(orientation="h", y=-0.05))
    return _base_layout(fig, 340, "Risks by Severity")


def stage_bar(matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for sev in SEVERITY_ORDER:
        fig.add_bar(name=sev, x=matrix.index, y=matrix[sev], marker_color=SEVERITY_COLORS[sev])
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(title="Number of risks", dtick=1)
    return _base_layout(fig, 380, "Risks per Lifecycle Stage (by severity)")


def category_bar(df: pd.DataFrame) -> go.Figure:
    counts = df["Risk Category"].value_counts().sort_values()
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color="#5B5BD6"))
    fig.update_xaxes(dtick=1, title="Number of risks")
    return _base_layout(fig, 380, "Risks by Category")


def rmf_bar(counts: dict) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=RMF_FUNCTIONS, y=[counts.get(f, 0) for f in RMF_FUNCTIONS],
        marker_color=[RMF_COLORS[f] for f in RMF_FUNCTIONS], text=[counts.get(f, 0) for f in RMF_FUNCTIONS],
        textposition="outside",
    ))
    fig.update_yaxes(title="Number of risks", dtick=1)
    return _base_layout(fig, 340, "Risks by NIST AI RMF Function")


def similarity_bar(results: list) -> go.Figure:
    """Bar chart of cosine similarity for the retrieved guidance of one risk."""
    ids = [f"{r['id']}" for r in results][::-1]
    sims = [r["similarity"] for r in results][::-1]
    cols = [RMF_COLORS[r["function"]] for r in results][::-1]
    fig = go.Figure(go.Bar(x=sims, y=ids, orientation="h", marker_color=cols,
                           text=[f"{s:.3f}" for s in sims], textposition="outside"))
    fig.update_xaxes(title="Cosine similarity (Sentence-BERT)", range=[0, max(1.0, max(sims) + 0.1)])
    return _base_layout(fig, 260, "Retrieved guidance: similarity scores")
