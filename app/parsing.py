from io import BytesIO
from pathlib import Path

from app.models import DocumentInput


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


class UnsupportedDocumentError(ValueError):
    pass


def parse_document(filename: str, content: bytes) -> list[DocumentInput]:
    safe_name = Path(filename).name
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentError(f"Unsupported file type '{extension}'. Allowed: {allowed}")
    if extension == ".pdf":
        return _parse_pdf(safe_name, content)
    if extension == ".docx":
        return _parse_docx(safe_name, content)
    return _parse_plain_text(safe_name, content)


def _parse_plain_text(filename: str, content: bytes) -> list[DocumentInput]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{filename} must be UTF-8 encoded") from exc
    if not text.strip():
        raise ValueError(f"{filename} contains no readable text")
    return [DocumentInput(source=filename, title=Path(filename).stem, text=text)]


def _parse_pdf(filename: str, content: bytes) -> list[DocumentInput]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not read PDF {filename}") from exc
    title = str((reader.metadata or {}).get("/Title") or Path(filename).stem)
    units = [
        DocumentInput(source=filename, title=title, page=index, text=text)
        for index, page in enumerate(reader.pages, start=1)
        if (text := (page.extract_text() or "").strip())
    ]
    if not units:
        raise ValueError(
            f"{filename} contains no extractable text; scanned PDFs require OCR"
        )
    return units


def _parse_docx(filename: str, content: bytes) -> list[DocumentInput]:
    from docx import Document

    try:
        document = Document(BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not read DOCX {filename}") from exc

    title = str(document.core_properties.title or Path(filename).stem)
    current_section: str | None = None
    section_paragraphs: list[str] = []
    units: list[DocumentInput] = []

    def flush() -> None:
        if section_paragraphs:
            units.append(
                DocumentInput(
                    source=filename,
                    title=title,
                    section=current_section,
                    text="\n".join(section_paragraphs),
                )
            )
            section_paragraphs.clear()

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if paragraph.style and paragraph.style.name.lower().startswith("heading"):
            flush()
            current_section = text
        else:
            section_paragraphs.append(text)
    flush()
    if not units:
        raise ValueError(f"{filename} contains no readable paragraphs")
    return units
