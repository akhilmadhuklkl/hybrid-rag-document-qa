import pytest
from docx import Document
from io import BytesIO
from pypdf import PdfWriter

from app.parsing import UnsupportedDocumentError, parse_document


def test_plain_text_upload_is_converted_to_document():
    documents = parse_document("policies.txt", b"Employees receive 20 leave days.")
    assert len(documents) == 1
    assert documents[0].source == "policies.txt"
    assert documents[0].title == "policies"


def test_upload_filename_is_sanitized():
    documents = parse_document("../../policies.md", b"# Leave\nTwenty days")
    assert documents[0].source == "policies.md"


def test_unsupported_upload_is_rejected():
    with pytest.raises(UnsupportedDocumentError):
        parse_document("malware.exe", b"not really an executable")


def test_invalid_text_encoding_is_rejected():
    with pytest.raises(ValueError, match="UTF-8"):
        parse_document("broken.txt", b"\xff\xfe")


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="no readable text"):
        parse_document("empty.md", b"  \n")


def test_docx_headings_are_preserved_as_sections():
    document = Document()
    document.core_properties.title = "Handbook"
    document.add_heading("Leave", level=1)
    document.add_paragraph("Employees receive 20 days.")
    buffer = BytesIO()
    document.save(buffer)
    units = parse_document("handbook.docx", buffer.getvalue())
    assert units[0].title == "Handbook"
    assert units[0].section == "Leave"
    assert "20 days" in units[0].text


def test_malformed_pdf_is_rejected_with_clear_error():
    with pytest.raises(ValueError, match="Could not read PDF"):
        parse_document("broken.pdf", b"not a PDF")


def test_scanned_or_empty_pdf_explains_ocr_requirement():
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buffer)
    with pytest.raises(ValueError, match="scanned PDFs require OCR"):
        parse_document("scan.pdf", buffer.getvalue())
