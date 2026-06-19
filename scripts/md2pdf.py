"""Convert MiniCode markdown doc to PDF using fpdf2 (pure Python, no GTK).

Usage:
    python scripts/md2pdf.py "C:/Users/123/Desktop/MiniCode项目深度解析.md"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF


def _clean_markdown(text: str) -> str:
    """Strip markdown syntax while preserving structure."""
    # Code blocks: replace with placeholder
    text = re.sub(r"```.*?```", " [code block] ", text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    # Links: keep text, drop URL
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Bullet points
    text = re.sub(r"^[-*] ", "  * ", text, flags=re.MULTILINE)
    # Blockquotes
    text = re.sub(r"^> ", "  ", text, flags=re.MULTILINE)
    # Headings (keep text)
    text = re.sub(r"^#{1,4} ", "", text, flags=re.MULTILINE)
    # Table rows (skip)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Horizontal rules
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    return text


def convert(md_path: str, pdf_path: str) -> None:
    md_text = Path(md_path).read_text(encoding="utf-8")
    cleaned = _clean_markdown(md_text)

    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=20)

    # Try CJK font, fall back to built-in
    font_name = "Helvetica"
    font_size = 10
    try:
        pdf.add_font("CJK", "", r"C:\Windows\Fonts\msyh.ttc")
        font_name = "CJK"
    except Exception:
        pass

    pdf.add_page()

    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]

    for para in paragraphs:
        lines = para.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Heading detection (all caps short lines or lines starting with keywords)
            if len(line) < 80 and (
                line.startswith("MiniCode")
                or line.startswith("AI ")
                or line.startswith("Part")
                or line.startswith("##")
                or line.startswith("核心")
                or line.startswith("项目")
                or line.startswith("技术")
                or line.startswith("附录")
                or line.startswith("Benchmark")
                or line.startswith("运行方式")
            ):
                pdf.set_font(font_name, "", 14)
                pdf.ln(6)
                pdf.cell(0, 10, line, ln=True)
                pdf.ln(2)
            elif len(line) < 100 and (
                line.startswith("2.")
                or line.startswith("--")
                or line.startswith("==")
            ):
                pdf.set_font(font_name, "", 12)
                pdf.ln(4)
                pdf.cell(0, 8, line, ln=True)
                pdf.ln(2)
            else:
                pdf.set_font(font_name, "", font_size)
                pdf.multi_cell(0, 6, line)

        pdf.ln(3)

    pdf.output(pdf_path)
    print(f"PDF saved to: {pdf_path}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "MiniCode项目深度解析.md"
    dst = sys.argv[2] if len(sys.argv) > 2 else "MiniCode项目深度解析.pdf"
    convert(src, dst)
