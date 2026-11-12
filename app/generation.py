from collections.abc import Sequence
import re
from typing import Protocol

from app.store import RetrievedChunk


class AnswerGenerator(Protocol):
    def generate(self, question: str, chunks: Sequence[RetrievedChunk]) -> str: ...


class ExtractiveGenerator:
    def generate(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        del question
        if not chunks:
            return "I couldn't find relevant information in the indexed documents."
        passages = [f"{chunk.text.strip()} [{index}]" for index, chunk in enumerate(chunks, 1)]
        return " ".join(passages)


class OpenAIGenerator:
    def __init__(self, model: str, api_key: str | None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        if not chunks:
            return "I couldn't find relevant information in the indexed documents."
        context = "\n\n".join(
            f"[{index}] Source: {chunk.source}"
            f"{f', page {chunk.page}' if chunk.page else ''}"
            f"{f', section {chunk.section}' if chunk.section else ''}\n{chunk.text}"
            for index, chunk in enumerate(chunks, 1)
        )
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "Answer only from the supplied company-document passages. "
                "Cite factual claims with [n]. If the passages do not contain the answer, say so."
            ),
            input=f"Question: {question}\n\nPassages:\n{context}",
        )
        return response.output_text


CITATION_PATTERN = re.compile(r"\[(\d+)]")


def has_valid_citations(answer: str, citation_count: int) -> bool:
    """Require at least one citation and reject references outside the context."""
    citations = [int(value) for value in CITATION_PATTERN.findall(answer)]
    return bool(citations) and all(1 <= value <= citation_count for value in citations)

