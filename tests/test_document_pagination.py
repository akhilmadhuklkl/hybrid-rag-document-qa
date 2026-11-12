from types import SimpleNamespace

from app.main import list_documents
from app.models import DocumentSummary


def test_document_listing_supports_pagination_and_totals():
    documents = [
        DocumentSummary(document_id=str(index), source=f"doc-{index}.md", chunk_count=index + 1, page_count=0, sections=[])
        for index in range(3)
    ]
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(store=SimpleNamespace(list_documents=lambda _owner: documents))),
        headers={},
    )
    response = list_documents(request, limit=1, offset=1)
    assert [item.document_id for item in response.documents] == ["1"]
    assert response.total_documents == 3
    assert response.total_chunks == 6
