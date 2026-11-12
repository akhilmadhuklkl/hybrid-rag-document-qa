from collections.abc import Sequence
from dataclasses import replace
import math
from typing import Any, Protocol

from app.store import RetrievedChunk


class Reranker(Protocol):
    def rerank(
        self, question: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class HuggingFaceCrossEncoder:
    """Minimal Transformers cross-encoder without optional sklearn imports."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def predict(
        self, pairs: list[tuple[str, str]], show_progress_bar: bool = False
    ) -> list[float]:
        del show_progress_bar
        questions = [question for question, _ in pairs]
        passages = [passage for _, passage in pairs]
        encoded = self.tokenizer(
            questions,
            passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with self.torch.no_grad():
            logits = self.model(**encoded).logits.reshape(-1)
        return [float(score) for score in logits.cpu().tolist()]


class CrossEncoderReranker:
    """Score question-passage pairs jointly with a cross-encoder model."""

    def __init__(self, model_name: str, model: Any | None = None):
        if model is None:
            model = HuggingFaceCrossEncoder(model_name)
        self.model = model

    def rerank(
        self, question: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(question, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)
        scored = [
            replace(
                chunk,
                score=float(score),
                evidence_score=1.0 / (1.0 + math.exp(-float(score))),
            )
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        return sorted(scored, key=lambda chunk: (-chunk.score, chunk.chunk_id))[:top_k]
