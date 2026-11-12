from app.reranking import CrossEncoderReranker
from app.store import RetrievedChunk


class FakeCrossEncoder:
    def predict(self, pairs, show_progress_bar=False):
        assert show_progress_bar is False
        return [0.1 if "irrelevant" in passage else 0.9 for _, passage in pairs]


def make_chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source="policy.md",
        title=None,
        page=None,
        section=None,
        score=0.01,
        evidence_score=0.1,
    )


def test_cross_encoder_reranks_and_limits_results():
    reranker = CrossEncoderReranker("unused", model=FakeCrossEncoder())
    results = reranker.rerank(
        "What is the policy?",
        [make_chunk("a", "irrelevant text"), make_chunk("b", "policy answer")],
        top_k=1,
    )
    assert [result.chunk_id for result in results] == ["b"]
    assert results[0].score == 0.9
    assert 0.7 < results[0].evidence_score < 0.8
