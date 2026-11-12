from app.models import DocumentSummary


def test_document_summary_tracks_source_structure():
    summary = DocumentSummary(
        document_id="abc123",
        source="handbook.pdf",
        title="Handbook",
        chunk_count=12,
        page_count=3,
        sections=["Leave", "Security"],
        ingested_at="2026-11-12T10:00:00+00:00",
    )
    assert summary.document_id == "abc123"
    assert summary.page_count == 3
    assert summary.sections == ["Leave", "Security"]
