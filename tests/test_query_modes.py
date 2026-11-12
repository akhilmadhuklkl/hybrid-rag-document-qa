import pytest
from pydantic import ValidationError

from app.models import QueryRequest


def test_query_mode_defaults_to_hybrid_and_accepts_keyword():
    assert QueryRequest(question="What is the policy?").mode == "hybrid"
    assert QueryRequest(question="What is the policy?", mode="keyword").mode == "keyword"


def test_query_mode_rejects_unknown_values():
    with pytest.raises(ValidationError):
        QueryRequest(question="What is the policy?", mode="lexical")
