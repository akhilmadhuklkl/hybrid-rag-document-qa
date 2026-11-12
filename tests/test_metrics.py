from app.store import RetrievedChunk
from evaluation.metrics import measure_query, summarize


def chunk(source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=source,
        text="text",
        source=source,
        title=None,
        page=None,
        section=None,
        score=1.0,
    )


def test_measure_query_calculates_reciprocal_rank():
    measurement = measure_query(
        "q1", [chunk("wrong.md"), chunk("right.md")], {"right.md"}, 12.0
    )
    assert measurement.hit is True
    assert measurement.reciprocal_rank == 0.5


def test_summarize_calculates_recall_and_mrr():
    hit = measure_query("q1", [chunk("right.md")], {"right.md"}, 10.0)
    miss = measure_query("q2", [chunk("wrong.md")], {"right.md"}, 20.0)
    report = summarize([hit, miss])
    assert report["recall_at_k"] == 0.5
    assert report["mrr"] == 0.5
    assert report["mean_latency_ms"] == 15.0


def test_summarize_calculates_nearest_rank_p95():
    measurements = [
        measure_query(f"q{index}", [], {"right.md"}, float(index))
        for index in range(1, 21)
    ]
    assert summarize(measurements)["p95_latency_ms"] == 19.0
