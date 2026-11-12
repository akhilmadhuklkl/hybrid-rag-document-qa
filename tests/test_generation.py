from app.generation import has_valid_citations


def test_grounded_answer_requires_citation():
    assert has_valid_citations("Employees receive 20 days [1].", 2)
    assert not has_valid_citations("Employees receive 20 days.", 2)


def test_grounded_answer_rejects_unknown_citation():
    assert not has_valid_citations("The policy says this [3].", 2)
