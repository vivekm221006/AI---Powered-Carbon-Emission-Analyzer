"""
PDF report generation using ReportLab.
"""

from datetime import datetime, timezone
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_GREEN = colors.HexColor("#2E7D32")
_LIGHT_GREEN = colors.HexColor("#E8F5E9")
_AMBER = colors.HexColor("#FFB300")
_WHITE = colors.white
_LIGHT_GREY = colors.HexColor("#F5F5F5")


def _make_table(data: list[list], col_widths: list[float]) -> Table:
    table = Table(data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _GREEN),
                ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 11),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_WHITE, _LIGHT_GREY]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
            ]
        )
    )
    return table


def generate_pdf_report(
    org_name: str,
    inputs: dict,
    total_co2: float,
    breakdown: dict,
    rating: tuple[str, str, str],
    recommendations: list[dict],
) -> BytesIO:
    """
    Generate a PDF report and return it as a BytesIO buffer.

    Parameters:
        org_name       — organisation name
        inputs         — dict with keys: electricity_kwh, fuel_liters, travel_km,
                         cloud_hours, waste_kg
        total_co2      — monthly total CO2 in tons
        breakdown      — {category: tons_co2} dict
        rating         — (letter, label, hex_color) tuple
        recommendations — list of recommendation dicts from recommendations.py
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=_GREEN,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    h1_style = ParagraphStyle(
        "H1Green",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=_GREEN,
        spaceBefore=12,
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=8,
        spaceAfter=2,
    )
    normal_style = styles["Normal"]
    normal_style.fontSize = 10

    rating_letter, rating_label, _ = rating
    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("🌍 Carbon Footprint Analysis Report", title_style))
    story.append(Paragraph(f"<b>Organisation:</b> {org_name}", styles["Normal"]))
    story.append(
        Paragraph(
            f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.25 * inch))

    # ── Executive Summary ─────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", h1_style))
    summary_data = [
        ["Metric", "Value"],
        ["Total Monthly CO₂ Emissions", f"{total_co2:.3f} tons CO₂"],
        ["Annual CO₂ Estimate", f"{total_co2 * 12:.2f} tons CO₂/year"],
        ["Carbon Rating", f"{rating_letter} — {rating_label}"],
    ]
    story.append(_make_table(summary_data, [3.25 * inch, 3.25 * inch]))
    story.append(Spacer(1, 0.2 * inch))

    # ── Input Data ────────────────────────────────────────────────────────────
    story.append(Paragraph("Organisation Input Data (Monthly)", h1_style))
    input_data = [
        ["Parameter", "Value"],
        ["Electricity Usage", f"{inputs['electricity_kwh']:,.0f} kWh"],
        ["Fuel Consumption", f"{inputs['fuel_liters']:,.0f} litres"],
        ["Business Travel Distance", f"{inputs['travel_km']:,.0f} km"],
        ["Cloud / Server Hours", f"{inputs['cloud_hours']:,.0f} hours"],
        ["Office Waste", f"{inputs['waste_kg']:,.0f} kg"],
    ]
    story.append(_make_table(input_data, [3.25 * inch, 3.25 * inch]))
    story.append(Spacer(1, 0.2 * inch))

    # ── Emission Breakdown ────────────────────────────────────────────────────
    story.append(Paragraph("CO₂ Emission Breakdown", h1_style))
    total_for_pct = sum(breakdown.values()) or 1
    bd_data = [["Category", "CO₂ (tons)", "Share (%)"]]
    for category, value in breakdown.items():
        bd_data.append(
            [category, f"{value:.4f}", f"{value / total_for_pct * 100:.1f}%"]
        )
    bd_data.append(["TOTAL", f"{total_co2:.4f}", "100.0%"])
    story.append(_make_table(bd_data, [3.0 * inch, 1.75 * inch, 1.75 * inch]))
    story.append(Spacer(1, 0.2 * inch))

    # ── Recommendations ───────────────────────────────────────────────────────
    if recommendations:
        story.append(Paragraph("AI Recommendations", h1_style))
        for rec in recommendations:
            story.append(Paragraph(rec["issue"], h2_style))
            story.append(
                Paragraph(f"<b>Status:</b> {rec.get('value', '')}", normal_style)
            )
            story.append(
                Paragraph(
                    f"<b>Priority:</b> {rec['priority']} &nbsp;|&nbsp; "
                    f"<b>Potential Reduction:</b> {rec['potential_reduction']}",
                    normal_style,
                )
            )
            story.append(Paragraph("<b>Suggested Actions:</b>", normal_style))
            for suggestion in rec["suggestions"]:
                story.append(Paragraph(f"&nbsp;&nbsp;• {suggestion}", normal_style))
            story.append(Spacer(1, 0.1 * inch))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3 * inch))
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )
    story.append(
        Paragraph(
            "Generated by AI-Powered Carbon Emission Analyzer · For internal use only",
            footer_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer
