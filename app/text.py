import hashlib
import re


TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, len(words), step)]


def chunk_id(
    source: str,
    position: int,
    text: str,
    page: int | None = None,
    section: str | None = None,
) -> str:
    identity = f"{source}:{page}:{section}:{position}:{text}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return f"{digest}-{position}"

