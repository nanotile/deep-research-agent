"""
PDF export utility for Research Agent Hub.
Converts markdown reports to downloadable PDF files.
Uses fpdf2 for pure-Python PDF generation (no system dependencies).
"""

import os
import re
import tempfile
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import fpdf2, graceful fallback if not available
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("fpdf2 not installed. PDF export will be disabled.")


class ResearchReportPDF(FPDF):
    """Custom PDF class for research reports with headers/footers."""

    def __init__(self, title: str = "Research Report"):
        super().__init__()
        self.report_title = title
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        """Add header to each page."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, self.report_title, align="L")
        self.cell(0, 10, datetime.now().strftime("%Y-%m-%d"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, 20, 200, 20)
        self.ln(5)

    def footer(self):
        """Add footer with page number."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting for plain text rendering."""
    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove images
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'[Image: \1]', text)
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '[Code Block]', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    return text


def _parse_markdown_sections(markdown: str) -> list:
    """Parse markdown into sections with formatting hints."""
    sections = []
    lines = markdown.split('\n')

    for line in lines:
        stripped = line.strip()

        if not stripped:
            sections.append(('blank', ''))
        elif stripped.startswith('# '):
            sections.append(('h1', stripped[2:]))
        elif stripped.startswith('## '):
            sections.append(('h2', stripped[3:]))
        elif stripped.startswith('### '):
            sections.append(('h3', stripped[4:]))
        elif stripped.startswith('#### '):
            sections.append(('h4', stripped[5:]))
        elif stripped.startswith('- ') or stripped.startswith('* '):
            sections.append(('bullet', stripped[2:]))
        elif re.match(r'^\d+\.\s', stripped):
            sections.append(('numbered', re.sub(r'^\d+\.\s', '', stripped)))
        elif stripped.startswith('|') and stripped.endswith('|'):
            sections.append(('table_row', stripped))
        elif stripped.startswith('---') or stripped.startswith('***'):
            sections.append(('hr', ''))
        elif stripped.startswith('>'):
            sections.append(('quote', stripped[1:].strip()))
        else:
            sections.append(('text', stripped))

    return sections


def markdown_to_pdf(
    markdown_content: str,
    title: str = "Research Report",
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Convert markdown report to PDF.

    Args:
        markdown_content: The markdown text to convert
        title: Title for the PDF header
        output_path: Optional path to save PDF. If None, creates temp file.

    Returns:
        Path to the generated PDF file, or None if PDF generation unavailable
    """
    if not PDF_AVAILABLE:
        logger.error("PDF generation not available. Install fpdf2: pip install fpdf2")
        return None

    if not markdown_content or not markdown_content.strip():
        logger.error("Empty markdown content provided")
        return None

    try:
        # Create PDF
        pdf = ResearchReportPDF(title=title)
        pdf.alias_nb_pages()
        pdf.add_page()

        # Parse markdown sections
        sections = _parse_markdown_sections(markdown_content)

        # Track if we're in a table
        in_table = False
        table_rows = []

        for section_type, content in sections:
            # Clean content of markdown formatting
            clean_content = _strip_markdown(content)

            if section_type == 'h1':
                if in_table:
                    _render_table(pdf, table_rows)
                    table_rows = []
                    in_table = False
                pdf.set_font("Helvetica", "B", 16)
                pdf.set_text_color(0, 51, 102)
                pdf.ln(5)
                pdf.multi_cell(0, 8, clean_content)
                pdf.ln(3)

            elif section_type == 'h2':
                if in_table:
                    _render_table(pdf, table_rows)
                    table_rows = []
                    in_table = False
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_text_color(0, 51, 102)
                pdf.ln(4)
                pdf.multi_cell(0, 7, clean_content)
                pdf.ln(2)

            elif section_type == 'h3':
                if in_table:
                    _render_table(pdf, table_rows)
                    table_rows = []
                    in_table = False
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_text_color(51, 51, 51)
                pdf.ln(3)
                pdf.multi_cell(0, 6, clean_content)
                pdf.ln(2)

            elif section_type == 'h4':
                if in_table:
                    _render_table(pdf, table_rows)
                    table_rows = []
                    in_table = False
                pdf.set_font("Helvetica", "BI", 11)
                pdf.set_text_color(51, 51, 51)
                pdf.ln(2)
                pdf.multi_cell(0, 6, clean_content)
                pdf.ln(1)

            elif section_type == 'bullet':
                if in_table:
                    _render_table(pdf, table_rows)
                    table_rows = []
                    in_table = False
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(5)  # Indent
                pdf.multi_cell(0, 5, f"• {clean_content}")

            elif section_type == 'numbered':
                if in_table:
                    _render_table(pdf, table_rows)
                    table_rows = []
                    in_table = False
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(5)  # Indent
                pdf.multi_cell(0, 5, clean_content)

            elif section_type == 'table_row':
                in_table = True
                # Parse table cells
                cells = [c.strip() for c in content.split('|')[1:-1]]
                # Skip separator rows (contain only dashes)
                if not all(re.match(r'^[-:]+$', c) for c in cells):
                    table_rows.append(cells)

            elif section_type == 'hr':
                if in_table:
                    _render_table(pdf, table_rows)
                    table_rows = []
                    in_table = False
                pdf.ln(3)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(3)

            elif section_type == 'quote':
                if in_table:
                    _render_table(pdf, table_rows)
                    table_rows = []
                    in_table = False
                pdf.set_font("Helvetica", "I", 10)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(10)  # Indent
                pdf.multi_cell(0, 5, clean_content)
                pdf.set_text_color(0, 0, 0)

            elif section_type == 'text' and clean_content:
                if in_table:
                    _render_table(pdf, table_rows)
                    table_rows = []
                    in_table = False
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 5, clean_content)

            elif section_type == 'blank':
                if in_table:
                    _render_table(pdf, table_rows)
                    table_rows = []
                    in_table = False
                pdf.ln(2)

        # Render any remaining table
        if in_table and table_rows:
            _render_table(pdf, table_rows)

        # Determine output path
        if output_path is None:
            # Create temp file
            fd, output_path = tempfile.mkstemp(suffix='.pdf', prefix='research_report_')
            os.close(fd)

        # Save PDF
        pdf.output(output_path)
        logger.info(f"PDF generated: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        return None


def _render_table(pdf: FPDF, rows: list):
    """Render a markdown table as PDF table."""
    if not rows:
        return

    pdf.ln(3)
    pdf.set_font("Helvetica", "", 9)

    # Calculate column widths
    num_cols = max(len(row) for row in rows)
    page_width = 190  # Available width
    col_width = page_width / num_cols

    # First row is header
    if rows:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(240, 240, 240)
        for cell in rows[0]:
            clean_cell = _strip_markdown(cell)
            pdf.cell(col_width, 6, clean_cell[:30], border=1, fill=True)
        pdf.ln()

        # Data rows
        pdf.set_font("Helvetica", "", 9)
        for row in rows[1:]:
            for i, cell in enumerate(row):
                clean_cell = _strip_markdown(cell)
                pdf.cell(col_width, 5, clean_cell[:30], border=1)
            pdf.ln()

    pdf.ln(2)


def generate_report_filename(ticker_or_query: str, report_type: str = "research") -> str:
    """Generate a filename for the report PDF."""
    # Clean the input for filename
    clean_name = re.sub(r'[^\w\s-]', '', ticker_or_query)[:30]
    clean_name = clean_name.replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{report_type}_{clean_name}_{timestamp}.pdf"
