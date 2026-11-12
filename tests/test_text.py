import pytest

from app.text import chunk_text, tokenize


def test_tokenize_is_case_insensitive_and_keeps_words():
    assert tokenize("Annual LEAVE: 20 days!") == ["annual", "leave", "20", "days"]


def test_chunk_text_applies_overlap():
    assert chunk_text("one two three four five", size=3, overlap=1) == [
        "one two three",
        "three four five",
        "five",
    ]


def test_chunk_text_rejects_non_progressing_window():
    with pytest.raises(ValueError):
        chunk_text("one two", size=2, overlap=2)

