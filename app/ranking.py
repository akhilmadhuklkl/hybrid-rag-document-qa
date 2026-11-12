from collections.abc import Iterable


def reciprocal_rank_fusion(
    rankings: Iterable[tuple[list[str], float]], rrf_k: int = 60
) -> list[tuple[str, float]]:
    """Merge ranked ID lists using weighted reciprocal-rank fusion."""
    scores: dict[str, float] = {}
    for ranked_ids, weight in rankings:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))

