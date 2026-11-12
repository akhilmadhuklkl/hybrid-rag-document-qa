from app.ranking import reciprocal_rank_fusion


def test_rrf_rewards_items_found_by_both_retrievers():
    result = reciprocal_rank_fusion(
        [(["semantic-only", "shared"], 0.6), (["shared", "keyword-only"], 0.4)],
        rrf_k=10,
    )
    assert result[0][0] == "shared"
    assert {item_id for item_id, _ in result} == {
        "shared",
        "semantic-only",
        "keyword-only",
    }

