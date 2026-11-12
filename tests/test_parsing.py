import pytest

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
