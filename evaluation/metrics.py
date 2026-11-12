from dataclasses import dataclass
from statistics import mean

from app.store import RetrievedChunk


@dataclass(frozen=True)
class QueryMeasurement:
    query_id: str
    reciprocal_rank: float
    hit: bool
    relevant_retrieved: int
    latency_ms: float


def measure_query(
    query_id: str,
    results: list[RetrievedChunk],
    expected_sources: set[str],
    latency_ms: float,
) -> QueryMeasurement:
    relevant_ranks = [
        rank
        for rank, result in enumerate(results, start=1)
        if result.source in expected_sources
    ]
    return QueryMeasurement(
        query_id=query_id,
        reciprocal_rank=1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        hit=bool(relevant_ranks),
        relevant_retrieved=len(relevant_ranks),
        latency_ms=latency_ms,
    )


def summarize(measurements: list[QueryMeasurement]) -> dict[str, float | int]:
    if not measurements:
        return {
            "queries": 0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "mean_relevant_chunks": 0.0,
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    latencies = sorted(item.latency_ms for item in measurements)
    p95_index = min(len(latencies) - 1, max(0, int(len(latencies) * 0.95 + 0.999) - 1))
    return {
        "queries": len(measurements),
        "recall_at_k": mean(float(item.hit) for item in measurements),
        "mrr": mean(item.reciprocal_rank for item in measurements),
        "mean_relevant_chunks": mean(item.relevant_retrieved for item in measurements),
        "mean_latency_ms": mean(item.latency_ms for item in measurements),
        "p95_latency_ms": latencies[p95_index],
    }
