from app.models import QueryResponse


def test_query_response_exposes_retrieval_metadata_defaults():
    response = QueryResponse(answer="answer", citations=[], conversation_id="conversation")
    assert response.retrieval_mode == "hybrid"
    assert response.retrieved_chunks == 0
    assert response.latency_ms == 0.0
