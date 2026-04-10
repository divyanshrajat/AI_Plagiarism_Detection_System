"""Generate a simple plain-text report for a submitted file.

This module produces a readable text report and is used by the web
application as a fallback/simple report format.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json

try:
    # optional PDF generation
    from reportlab.lib.pagesizes import A4  # type: ignore
    from reportlab.pdfgen import canvas  # type: ignore
    _HAVE_REPORTLAB = True
except ImportError:
    _HAVE_REPORTLAB = False
# Some functions here are intentionally compact for readability; allow
# slightly higher complexity for report formatting and metadata handling.
# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments


def _generate_pdf(lines: List[str], pdf_path: Path) -> bool:
    try:
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4
        y = height - 72
        # attempt to draw logo if present in statics
        logo_path = Path(__file__).resolve().parents[1] / 'statics' / 'logo.png'
        if logo_path.exists():
            try:
                c.drawImage(str(logo_path), 72, height - 100, width=80, preserveAspectRatio=True, mask='auto')
            except Exception:
                pass
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72 + 90, y, "AI-Based Plagiarism Detection Report")
        y -= 28
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
        y -= 20
        c.setFont("Helvetica", 10)
        # render lines with simple word-wrapping
        for line in lines:
            text = c.beginText(72, y)
            for chunk in _wrap_text(line, int((width - 144) / 6)):
                text.textLine(chunk)
                y -= 12
                if y < 72:
                    c.drawText(text)
                    c.showPage()
                    y = height - 72
                    text = c.beginText(72, y)
            c.drawText(text)
        c.save()
        return True
    except OSError:
        return False


def _wrap_text(s: str, max_chars: int):
    if not s:
        return ['']
    words = s.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + (1 if cur else 0) > max_chars:
            lines.append(' '.join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += len(w) + (1 if cur_len else 0)
    if cur:
        lines.append(' '.join(cur))
    return lines


def generate_report(
    original_filename: str,
    plagiarism_score: float,
    top_matches: List[Dict[str, Any]],
    ai_indicator: Dict[str, Any],
    report_dir: Path,
    owner: str | None = None,
) -> tuple[Path, Path | None]:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"report_{timestamp}.txt"

    lines = [
        "AI-Based Plagiarism Detection Report",
        "=" * 40,
        f"Submitted file: {original_filename}",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Overall plagiarism score: {plagiarism_score:.2f}%",
        "",
        "Top similar references:",
    ]

    if top_matches:
        for idx, match in enumerate(top_matches, start=1):
            source = match.get("source", "<unknown>")
            try:
                score_val = float(match.get("score", 0.0))
            except (TypeError, ValueError):
                score_val = 0.0
            lines.append(f"  {idx}. {source} -> {score_val:.2f}%")
    else:
        lines.append("  No references available.")

    lines += [
        "",
        "AI-assisted text indicator:",
        f"  Probability: {float(ai_indicator.get('probability', 0.0)):.2f}%",
        f"  Label: {ai_indicator.get('label', 'N/A')}",
        f"  Note: {ai_indicator.get('explanation', '')}",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Save a small index of reports for teacher dashboard / exports
    idx_file = report_dir / "index.json"
    try:
        if idx_file.exists():
            existing = json.loads(idx_file.read_text(encoding="utf-8") or "[]")
        else:
            existing = []
        entry = {
            "path": report_path.name,
            "timestamp": timestamp,
            "score": float(plagiarism_score),
        }
        if owner:
            entry["owner"] = owner
        existing.insert(0, entry)
        idx_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        pass

    # Optionally create a simple PDF version if reportlab is available
    pdf_path = None
    if _HAVE_REPORTLAB:
        candidate_pdf = report_dir / f"report_{timestamp}.pdf"
        if _generate_pdf(lines, candidate_pdf):
            pdf_path = candidate_pdf

    return report_path, pdf_path
