import pytest
from fastapi import HTTPException
from types import SimpleNamespace

from app.main import get_document
from app.models import DocumentSummary


def test_document_detail_is_owner_scoped():
    expected = DocumentSummary(document_id="abc", source="policy.md", chunk_count=1, page_count=0, sections=[])
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        store=SimpleNamespace(list_documents=lambda owner: [expected] if owner == "alice" else []),
    )))
    request.headers = {"x-user-id": "alice"}
    assert get_document("abc", request) == expected


def test_document_detail_returns_404_for_missing():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        store=SimpleNamespace(list_documents=lambda _owner: []),
    )), headers={})
    with pytest.raises(HTTPException) as error:
        get_document("missing", request)
    assert error.value.status_code == 404
