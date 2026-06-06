"""PDF clustering-report exporter.

Produces a clean, print-friendly PDF summarizing a session's clustering result:
headline metrics, a cluster summary table, and per-cluster requirement listings
(capped per cluster to keep the document usable at scale). Light theme with teal
accents - meant to be printed/shared, not a dark on-screen deck.

Pure-Python (reportlab); no system dependencies.
"""

from __future__ import annotations

import io
from collections import defaultdict
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Qualitative palette for charts (teal-led, print-friendly, no purple).
_CHART_COLORS = [
    "#0D8175", "#2FBCAA", "#0EA5E9", "#F59E0B", "#84CC16", "#EC4899",
    "#14B8A6", "#3B82F6", "#D97706", "#22C55E", "#E11D48", "#0891B2",
]

TEAL = colors.HexColor("#0D8175")
INK = colors.HexColor("#0F1A18")
MUTED = colors.HexColor("#5B6B68")
LINE = colors.HexColor("#D7E0DE")
NOISE = colors.HexColor("#9AA7A4")

# Cap requirements listed per cluster so a 50k-requirement report stays readable.
MAX_PER_CLUSTER = 25


def _esc(value: Any) -> str:
    return (
        str(value if value is not None else "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    return {
        "title": ParagraphStyle("title", parent=base, fontName="Helvetica-Bold",
                                fontSize=20, textColor=INK, leading=24, spaceAfter=2),
        "sub": ParagraphStyle("sub", parent=base, fontSize=9.5, textColor=MUTED, spaceAfter=2),
        "metric_num": ParagraphStyle("mn", parent=base, fontName="Helvetica-Bold",
                                     fontSize=18, textColor=TEAL, leading=20),
        "metric_lbl": ParagraphStyle("ml", parent=base, fontSize=7.5, textColor=MUTED, leading=9),
        "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=13,
                             textColor=TEAL, leading=16, spaceBefore=12, spaceAfter=6),
        "clab": ParagraphStyle("clab", parent=base, fontName="Helvetica-Bold", fontSize=11.5,
                               textColor=INK, leading=14, spaceBefore=10, spaceAfter=1),
        "kw": ParagraphStyle("kw", parent=base, fontSize=8.5, textColor=TEAL, leading=11, spaceAfter=3),
        "cellh": ParagraphStyle("cellh", parent=base, fontName="Helvetica-Bold", fontSize=8.5,
                                textColor=colors.white, leading=11),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.5, textColor=INK, leading=11),
        "cellm": ParagraphStyle("cellm", parent=base, fontSize=8.5, textColor=MUTED, leading=11),
        "req": ParagraphStyle("req", parent=base, fontSize=8.5, textColor=INK, leading=11.5, spaceAfter=2,
                              leftIndent=4),
    }


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(16 * mm, 10 * mm, "ReqCluster - Clustering Report")
    canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.line(16 * mm, 12 * mm, A4[0] - 16 * mm, 12 * mm)
    canvas.restoreState()


def _new_figure(width_in, height_in):
    """Lazily create a matplotlib figure (Agg). Returns (fig, ax, plt) or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=150)
    return fig, ax, plt


def _fig_to_buf(fig, plt) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _scatter_png(requirements: List[Dict[str, Any]]) -> io.BytesIO | None:
    """UMAP 2-d scatter coloured by cluster (noise in grey). None if no coords."""
    pts = [
        (r.get("umap_x"), r.get("umap_y"), r.get("cluster_id"))
        for r in requirements
        if r.get("umap_x") is not None and r.get("umap_y") is not None
    ]
    if len(pts) < 3:
        return None
    made = _new_figure(6.6, 3.3)
    if made is None:
        return None
    fig, ax, plt = made
    uniq = sorted({c for _, _, c in pts if c is not None and c != -1})
    cmap = {c: _CHART_COLORS[i % len(_CHART_COLORS)] for i, c in enumerate(uniq)}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cols = ["#C9D2D0" if c == -1 else cmap.get(c, "#0D8175") for _, _, c in pts]
    ax.scatter(xs, ys, c=cols, s=7, alpha=0.75, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("UMAP projection (colored by cluster)", fontsize=9, color="#0F1A18")
    for spine in ax.spines.values():
        spine.set_color("#D7E0DE")
    return _fig_to_buf(fig, plt)


def _bar_png(clusters: List[Dict[str, Any]]) -> io.BytesIO | None:
    """Horizontal bar chart of the largest clusters. None if matplotlib missing."""
    top = sorted(clusters, key=lambda c: c.get("size", 0), reverse=True)[:15]
    if not top:
        return None
    made = _new_figure(6.6, max(2.2, 0.28 * len(top) + 0.6))
    if made is None:
        return None
    fig, ax, plt = made
    labels = [(str(c.get("label") or c.get("cluster_id")))[:26] for c in top]
    sizes = [c.get("size", 0) for c in top]
    ax.barh(range(len(top)), sizes, color="#0D8175")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7.5, color="#0F1A18")
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=7.5, colors="#5B6B68")
    ax.set_title("Largest clusters", fontsize=9, color="#0F1A18")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D7E0DE")
    return _fig_to_buf(fig, plt)


def export_pdf(data: Dict[str, Any]) -> bytes:
    reqs: List[Dict[str, Any]] = data.get("requirements", [])
    clusters: List[Dict[str, Any]] = data.get("clusters", [])
    session = data.get("session", {})
    deps: List[Dict[str, Any]] = data.get("dependencies", [])

    n = len(reqs)
    noise = sum(1 for r in reqs if r.get("is_noise"))
    coverage = round((n - noise) / n * 100, 1) if n else 0.0

    by_cluster: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in reqs:
        by_cluster[r.get("cluster_id")].append(r)
    ordered = sorted(clusters, key=lambda c: c.get("size", 0), reverse=True)
    avail = A4[0] - 32 * mm

    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, title="ReqCluster Clustering Report",
        topMargin=16 * mm, bottomMargin=18 * mm, leftMargin=16 * mm, rightMargin=16 * mm,
    )
    story: List[Any] = []

    # --- header ---
    story.append(Paragraph("ReqCluster - Clustering Report", st["title"]))
    story.append(Paragraph(
        f"Session: {_esc(session.get('name') or session.get('filename') or 'Untitled')}"
        f"  ·  Generated: {_esc(data.get('timestamp', ''))}", st["sub"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", color=LINE, thickness=1))
    story.append(Spacer(1, 8))

    # --- metric strip ---
    def metric(num, lbl):
        return [Paragraph(str(num), st["metric_num"]), Paragraph(lbl, st["metric_lbl"])]

    metrics = Table(
        [[
            metric(n, "REQUIREMENTS"),
            metric(len([c for c in clusters]), "CLUSTERS"),
            metric(noise, "NOISE"),
            metric(f"{coverage}%", "COVERAGE"),
        ]],
        colWidths=[(A4[0] - 32 * mm) / 4.0] * 4,
    )
    metrics.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(metrics)
    story.append(Spacer(1, 10))

    # --- charts (skipped gracefully if matplotlib is unavailable) ---
    scatter = _scatter_png(reqs)
    if scatter is not None:
        story.append(Paragraph("Cluster map", st["h2"]))
        story.append(Image(scatter, width=avail, height=avail * 0.5))
        story.append(Spacer(1, 8))
    bar = _bar_png(clusters)
    if bar is not None:
        story.append(Image(bar, width=avail * 0.78, height=avail * 0.78 * 0.5))
        story.append(Spacer(1, 6))

    # --- cluster summary table ---
    story.append(Paragraph("Cluster summary", st["h2"]))
    head = [Paragraph(h, st["cellh"]) for h in ("ID", "Label", "Size", "Top keywords")]
    rows = [head]
    for c in ordered:
        cid = c["cluster_id"]
        kws = ", ".join(str(k) for k in (c.get("keywords") or [])[:6])
        rows.append([
            Paragraph(_esc(cid), st["cell"]),
            Paragraph(_esc(c.get("label", "")), st["cell"]),
            Paragraph(_esc(c.get("size", 0)), st["cell"]),
            Paragraph(_esc(kws), st["cellm"]),
        ])
    summary = Table(rows, colWidths=[avail * 0.08, avail * 0.4, avail * 0.1, avail * 0.42], repeatRows=1)
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F7F6")]),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(summary)

    # --- per-cluster detail ---
    story.append(Paragraph("Clusters in detail", st["h2"]))
    for c in ordered:
        cid = c["cluster_id"]
        members = by_cluster.get(cid, [])
        block: List[Any] = [
            Paragraph(f"Cluster {_esc(cid)}: {_esc(c.get('label',''))}  "
                      f"<font color='#5B6B68'>({len(members)} requirements)</font>", st["clab"]),
        ]
        if c.get("keywords"):
            block.append(Paragraph("Keywords: " + _esc(", ".join(str(k) for k in c["keywords"])), st["kw"]))
        for r in members[:MAX_PER_CLUSTER]:
            block.append(Paragraph(
                f"<b>{_esc(r.get('req_id') or '')}</b>&nbsp;&nbsp;{_esc(r.get('text',''))}", st["req"]))
        if len(members) > MAX_PER_CLUSTER:
            block.append(Paragraph(
                f"... and {len(members) - MAX_PER_CLUSTER} more requirements in this cluster.", st["cellm"]))
        # Keep the heading with at least its first lines together.
        story.append(KeepTogether(block[:3]))
        for flow in block[3:]:
            story.append(flow)
        story.append(Spacer(1, 4))

    # --- noise ---
    noise_members = by_cluster.get(-1, [])
    if noise_members:
        story.append(Paragraph("Noise (unclustered)", st["h2"]))
        story.append(Paragraph(
            f"{len(noise_members)} requirements did not fit any cluster.", st["cellm"]))
        for r in noise_members[:MAX_PER_CLUSTER]:
            story.append(Paragraph(
                f"<b>{_esc(r.get('req_id') or '')}</b>&nbsp;&nbsp;{_esc(r.get('text',''))}", st["req"]))
        if len(noise_members) > MAX_PER_CLUSTER:
            story.append(Paragraph(f"... and {len(noise_members) - MAX_PER_CLUSTER} more.", st["cellm"]))

    # --- dependency summary (if generated) ---
    if deps:
        rel_counts: Dict[str, int] = defaultdict(int)
        for d in deps:
            rel_counts[d.get("relation", "other")] += 1
        story.append(Paragraph("Dependencies", st["h2"]))
        summary_txt = ", ".join(f"{v} {k}" for k, v in sorted(rel_counts.items()))
        story.append(Paragraph(
            f"{len(deps)} dependency links inferred ({_esc(summary_txt)}).", st["cellm"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
