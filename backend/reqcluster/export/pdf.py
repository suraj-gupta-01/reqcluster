"""Print-friendly PDF clustering report (reportlab; matplotlib chart optional)."""
from __future__ import annotations

import functools
import io
import re
from collections import defaultdict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .bundle import ExportBundle, xml_safe

TEAL = colors.HexColor("#0D8175")
INK = colors.HexColor("#0F1A18")
MUTED = colors.HexColor("#5B6B68")
LINE = colors.HexColor("#D7E0DE")
MAX_PER_CLUSTER = 25  # keeps a 50k-requirement report readable


def _esc(v) -> str:
    return xml_safe(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_CHART_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


@functools.lru_cache(maxsize=1)
def _chart_glyphs() -> frozenset[int]:
    from matplotlib import font_manager, ft2font

    return frozenset(ft2font.FT2Font(font_manager.findfont("DejaVu Sans")).get_charmap())


def _chart_label(text: str) -> str:
    """Chart text is drawn with DejaVu Sans: drop control chars, '?' for glyphs it lacks (else
    matplotlib draws empty boxes and warns per glyph), escape "$" (mathtext)."""
    glyphs = _chart_glyphs()
    clean = "".join(ch if ord(ch) in glyphs else "?" for ch in _CHART_CTRL.sub("", text))
    return clean.replace("$", r"\$")


def _bar_png(clusters):
    top = sorted(clusters, key=lambda c: c.size, reverse=True)[:15]
    if not top:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(6.6, max(2.2, 0.28 * len(top) + 0.6)), dpi=150)
    ax.barh(range(len(top)), [c.size for c in top], color="#0D8175")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([_chart_label(c.label[:26]) for c in top], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_title("Largest clusters", fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def export_pdf(b: ExportBundle) -> bytes:
    base = getSampleStyleSheet()["Normal"]
    S = {
        "title": ParagraphStyle("t", parent=base, fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=INK),
        "sub": ParagraphStyle("s", parent=base, fontSize=9.5, textColor=MUTED),
        "h2": ParagraphStyle("h", parent=base, fontName="Helvetica-Bold", fontSize=13, leading=16,
                             textColor=TEAL, spaceBefore=12, spaceAfter=6),
        "cell": ParagraphStyle("c", parent=base, fontSize=8.5, leading=11, textColor=INK),
        "muted": ParagraphStyle("m", parent=base, fontSize=8.5, leading=11, textColor=MUTED),
        "req": ParagraphStyle("r", parent=base, fontSize=8.5, leading=11.5, spaceAfter=2, leftIndent=4),
    }
    reqs = b.requirements
    n = len(reqs)
    noise = sum(r.is_noise for r in reqs)
    assigned = sum(r.cluster_id is not None and not r.is_noise for r in reqs)
    avail = A4[0] - 32 * mm
    by = defaultdict(list)
    for r in reqs:
        by[r.cluster_id].append(r)

    story = [
        Paragraph("ReqCluster - Clustering Report", S["title"]),
        Paragraph(f"Session: {_esc(b.session_name or 'Untitled')}  ·  Generated: {_esc(b.generated_at)}", S["sub"]),
        Spacer(1, 6), HRFlowable(width="100%", color=LINE, thickness=1), Spacer(1, 8),
        Paragraph(f"{n} requirements, {len(b.clusters)} clusters, {noise} noise, "
                  f"{round(assigned / n * 100, 1) if n else 0.0}% coverage", S["cell"]),
    ]
    bar = _bar_png(b.clusters)
    if bar is not None:
        story += [Spacer(1, 8), Image(bar, width=avail * 0.78, height=avail * 0.39)]

    ordered = sorted(b.clusters, key=lambda c: c.size, reverse=True)
    story.append(Paragraph("Cluster summary", S["h2"]))
    rows = [[Paragraph(h, S["cell"]) for h in ("ID", "Label", "Size", "Top keywords")]]
    for c in ordered:
        rows.append([Paragraph(_esc(c.id), S["cell"]), Paragraph(_esc(c.label), S["cell"]),
                     Paragraph(_esc(c.size), S["cell"]),
                     Paragraph(_esc(", ".join(map(str, c.keywords[:6]))), S["muted"])])
    t = Table(rows, colWidths=[avail * .08, avail * .4, avail * .1, avail * .42], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D7EFEC")),
                           ("GRID", (0, 0), (-1, -1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t)

    def listing(members):
        for r in members[:MAX_PER_CLUSTER]:
            story.append(Paragraph(f"<b>{_esc(r.req_id)}</b>&nbsp;&nbsp;{_esc(r.text)}", S["req"]))
        if len(members) > MAX_PER_CLUSTER:
            story.append(Paragraph(f"... and {len(members) - MAX_PER_CLUSTER} more.", S["muted"]))

    story.append(Paragraph("Clusters in detail", S["h2"]))
    for c in ordered:
        story.append(Paragraph(f"<b>Cluster {_esc(c.id)}: {_esc(c.label)}</b> ({len(by[c.id])} requirements)", S["cell"]))
        listing(by[c.id])
        story.append(Spacer(1, 4))
    for key, title in ((-1, "Noise (unclustered)"), (None, "Unclustered")):
        if by.get(key):
            story.append(Paragraph(title, S["h2"]))
            listing(by[key])

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title="ReqCluster Clustering Report", topMargin=16 * mm,
                      bottomMargin=18 * mm, leftMargin=16 * mm, rightMargin=16 * mm).build(story)
    return buf.getvalue()
