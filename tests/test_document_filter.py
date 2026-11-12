from types import SimpleNamespace

from app.main import list_documents
from app.models import DocumentSummary


def test_document_listing_filters_source_before_pagination():
    documents = [
        DocumentSummary(document_id="1", source="finance-policy.md", chunk_count=2, page_count=0, sections=[]),
        DocumentSummary(document_id="2", source="engineering-guide.md", chunk_count=3, page_count=0, sections=[]),
    ]
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(store=SimpleNamespace(list_documents=lambda _owner: documents))),
        headers={},
    )
    response = list_documents(request, limit=10, offset=0, source="FINANCE")
    assert [item.document_id for item in response.documents] == ["1"]
    assert response.total_documents == 1
    assert response.total_chunks == 2
