"""
Google Drive Service - Export reports to Google Drive as Google Docs

Uses Service Account authentication to save reports directly to a shared folder.
Converts markdown content to Google Docs format with proper styling.
"""

import os
import re
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DriveExportResult:
    """Result of a Google Drive export operation"""
    success: bool
    doc_id: Optional[str] = None
    doc_url: Optional[str] = None
    error: Optional[str] = None


def is_drive_configured() -> bool:
    """Check if Google Drive credentials are configured"""
    service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

    if not service_account_file or not folder_id:
        return False

    # Check if the service account file exists
    if not os.path.isfile(service_account_file):
        logger.warning(f"Service account file not found: {service_account_file}")
        return False

    return True


def get_credentials():
    """Load Google credentials from service account file"""
    from google.oauth2 import service_account

    SCOPES = [
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/documents'
    ]

    service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

    if not service_account_file:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_FILE not set in environment")

    if not os.path.isfile(service_account_file):
        raise FileNotFoundError(f"Service account file not found: {service_account_file}")

    credentials = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=SCOPES
    )

    return credentials


def markdown_to_doc_requests(markdown: str) -> list:
    """
    Convert markdown text to Google Docs API batchUpdate requests.

    Handles:
    - # Header → Heading 1
    - ## Header → Heading 2
    - ### Header → Heading 3
    - **bold** → Bold text
    - *italic* → Italic text
    - - item → Bullet list
    - [text](url) → Hyperlink
    - Plain text → Normal paragraph
    """
    requests = []
    current_index = 1  # Google Docs starts at index 1

    lines = markdown.split('\n')

    for line in lines:
        if not line.strip():
            # Empty line - add newline
            requests.append({
                'insertText': {
                    'location': {'index': current_index},
                    'text': '\n'
                }
            })
            current_index += 1
            continue

        # Determine heading level
        heading_style = None
        if line.startswith('### '):
            heading_style = 'HEADING_3'
            line = line[4:]
        elif line.startswith('## '):
            heading_style = 'HEADING_2'
            line = line[3:]
        elif line.startswith('# '):
            heading_style = 'HEADING_1'
            line = line[2:]

        # Check for bullet points
        is_bullet = False
        if line.startswith('- ') or line.startswith('* '):
            is_bullet = True
            line = line[2:]

        # Process inline formatting (bold, italic, links)
        text_segments = []
        formatting_requests = []

        # Parse the line for formatting
        segment_start = current_index
        processed_line, inline_formats = parse_inline_formatting(line, segment_start)

        # Insert the text
        text_to_insert = processed_line + '\n'
        requests.append({
            'insertText': {
                'location': {'index': current_index},
                'text': text_to_insert
            }
        })

        text_end = current_index + len(processed_line)

        # Apply heading style if applicable
        if heading_style:
            requests.append({
                'updateParagraphStyle': {
                    'range': {
                        'startIndex': current_index,
                        'endIndex': text_end + 1
                    },
                    'paragraphStyle': {
                        'namedStyleType': heading_style
                    },
                    'fields': 'namedStyleType'
                }
            })

        # Apply bullet formatting if applicable
        if is_bullet:
            requests.append({
                'createParagraphBullets': {
                    'range': {
                        'startIndex': current_index,
                        'endIndex': text_end + 1
                    },
                    'bulletPreset': 'BULLET_DISC_CIRCLE_SQUARE'
                }
            })

        # Apply inline formatting (bold, italic, links)
        for fmt in inline_formats:
            requests.append(fmt)

        current_index += len(text_to_insert)

    return requests


def parse_inline_formatting(text: str, start_index: int) -> tuple[str, list]:
    """
    Parse inline markdown formatting and return clean text plus formatting requests.

    Returns:
        tuple: (clean_text, list of formatting requests)
    """
    formatting_requests = []
    clean_text = text
    offset_adjustment = 0

    # Process links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    for match in re.finditer(link_pattern, text):
        link_text = match.group(1)
        link_url = match.group(2)

        # Calculate position in clean text
        original_start = match.start() - offset_adjustment
        original_end = original_start + len(link_text)

        # Replace the full markdown link with just the text
        full_match = match.group(0)
        clean_text = clean_text[:original_start + offset_adjustment] + link_text + clean_text[original_start + offset_adjustment + len(full_match):]

        # Add link formatting
        formatting_requests.append({
            'updateTextStyle': {
                'range': {
                    'startIndex': start_index + original_start,
                    'endIndex': start_index + original_end
                },
                'textStyle': {
                    'link': {'url': link_url}
                },
                'fields': 'link'
            }
        })

        # Adjust offset for subsequent matches
        offset_adjustment += len(full_match) - len(link_text)

    # Re-parse clean text for bold and italic
    # Process bold: **text**
    bold_pattern = r'\*\*([^*]+)\*\*'
    temp_text = clean_text
    offset = 0
    for match in re.finditer(bold_pattern, temp_text):
        bold_text = match.group(1)
        pos_in_clean = match.start() - offset

        # Replace **text** with text
        clean_text = clean_text[:pos_in_clean] + bold_text + clean_text[pos_in_clean + len(match.group(0)):]

        formatting_requests.append({
            'updateTextStyle': {
                'range': {
                    'startIndex': start_index + pos_in_clean,
                    'endIndex': start_index + pos_in_clean + len(bold_text)
                },
                'textStyle': {
                    'bold': True
                },
                'fields': 'bold'
            }
        })
        offset += 4  # Removed 4 asterisks

    # Process italic: *text* (but not **)
    italic_pattern = r'(?<!\*)\*([^*]+)\*(?!\*)'
    temp_text = clean_text
    offset = 0
    for match in re.finditer(italic_pattern, temp_text):
        italic_text = match.group(1)
        pos_in_clean = match.start() - offset

        clean_text = clean_text[:pos_in_clean] + italic_text + clean_text[pos_in_clean + len(match.group(0)):]

        formatting_requests.append({
            'updateTextStyle': {
                'range': {
                    'startIndex': start_index + pos_in_clean,
                    'endIndex': start_index + pos_in_clean + len(italic_text)
                },
                'textStyle': {
                    'italic': True
                },
                'fields': 'italic'
            }
        })
        offset += 2  # Removed 2 asterisks

    return clean_text, formatting_requests


def create_google_doc(title: str, markdown_content: str) -> DriveExportResult:
    """
    Create a Google Doc from markdown content and save to configured Drive folder.

    Args:
        title: Document title
        markdown_content: Markdown-formatted report content

    Returns:
        DriveExportResult with doc_id and doc_url on success, error on failure
    """
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

        if not is_drive_configured():
            return DriveExportResult(
                success=False,
                error="Google Drive not configured. Set GOOGLE_SERVICE_ACCOUNT_FILE and GOOGLE_DRIVE_FOLDER_ID in .env"
            )

        credentials = get_credentials()
        folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

        # Build Drive and Docs services
        drive_service = build('drive', 'v3', credentials=credentials)
        docs_service = build('docs', 'v1', credentials=credentials)

        # Step 1: Create empty Google Doc in the specified folder
        file_metadata = {
            'name': title,
            'mimeType': 'application/vnd.google-apps.document',
            'parents': [folder_id]
        }

        doc_file = drive_service.files().create(
            body=file_metadata,
            fields='id, webViewLink'
        ).execute()

        doc_id = doc_file.get('id')
        doc_url = doc_file.get('webViewLink')

        logger.info(f"Created Google Doc: {doc_id}")

        # Step 2: Convert markdown to Docs format and insert content
        requests = markdown_to_doc_requests(markdown_content)

        if requests:
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': requests}
            ).execute()

            logger.info(f"Inserted content into doc: {doc_id}")

        return DriveExportResult(
            success=True,
            doc_id=doc_id,
            doc_url=doc_url
        )

    except ImportError as e:
        logger.error(f"Google API libraries not installed: {e}")
        return DriveExportResult(
            success=False,
            error="Google API libraries not installed. Run: pip install google-auth google-api-python-client"
        )
    except HttpError as e:
        logger.error(f"Google API error: {e}")
        return DriveExportResult(
            success=False,
            error=f"Google API error: {e.reason}"
        )
    except Exception as e:
        logger.error(f"Error creating Google Doc: {e}")
        return DriveExportResult(
            success=False,
            error=str(e)
        )


def save_report_to_drive(report: str, query: str) -> str:
    """
    Save a research report to Google Drive.

    Args:
        report: The markdown report content
        query: The original research query (used for title)

    Returns:
        Status message for display in UI
    """
    if not report or report.startswith("*"):
        return "❌ No report to save"

    if not is_drive_configured():
        return "⚠️ Google Drive not configured. Set GOOGLE_SERVICE_ACCOUNT_FILE and GOOGLE_DRIVE_FOLDER_ID in .env"

    # Create a clean title from the query
    title = f"AI Research: {query[:50]}" if len(query) > 50 else f"AI Research: {query}"

    result = create_google_doc(title, report)

    if result.success:
        return f"✅ Saved to Google Drive!\n\n[📄 Open in Google Docs]({result.doc_url})"
    else:
        return f"❌ Failed to save: {result.error}"
