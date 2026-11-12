from dataclasses import dataclass
from typing import Any, Literal

import chromadb
from rank_bm25 import BM25Okapi

from app.config import Settings
from app.models import DocumentInput
from app.ranking import reciprocal_rank_fusion
from app.text import chunk_id, chunk_text, tokenize


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    title: str | None
    page: int | None
    section: str | None
    score: float


class HybridDocumentStore:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        self.client = client or chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, documents: list[DocumentInput]) -> int:
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for document in documents:
            chunks = chunk_text(
                document.text, self.settings.chunk_size, self.settings.chunk_overlap
            )
            for position, text in enumerate(chunks):
                ids.append(
                    chunk_id(
                        document.source,
                        position,
                        text,
                        page=document.page,
                        section=document.section,
                    )
                )
                texts.append(text)
                metadatas.append(
                    {
                        "source": document.source,
                        "title": document.title or "",
                        "position": position,
                        "page": document.page or 0,
                        "section": document.section or "",
                    }
                )
        if ids:
            self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        return len(ids)

    def search(
        self,
        question: str,
        top_k: int,
        mode: Literal["semantic", "keyword", "hybrid"] = "hybrid",
    ) -> list[RetrievedChunk]:
        stored = self.collection.get(include=["documents", "metadatas"])
        all_ids = stored.get("ids", [])
        all_texts = stored.get("documents") or []
        all_metadata = stored.get("metadatas") or []
        if not all_ids:
            return []

        candidate_count = min(len(all_ids), max(top_k * 4, top_k))
        semantic_ids: list[str] = []
        keyword_ids: list[str] = []
        if mode in {"semantic", "hybrid"}:
            semantic = self.collection.query(
                query_texts=[question], n_results=candidate_count, include=["documents"]
            )
            semantic_ids = (semantic.get("ids") or [[]])[0]

        if mode in {"keyword", "hybrid"}:
            # BM25 expects a non-empty average document length. Keep unusual inputs
            # such as punctuation-only documents searchable by the semantic branch.
            corpus_tokens = [tokenize(text) or [""] for text in all_texts]
            bm25 = BM25Okapi(corpus_tokens)
            query_tokens = tokenize(question)
            keyword_scores = (
                bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(all_ids)
            )
            keyword_order = sorted(
                range(len(all_ids)),
                key=lambda index: (-keyword_scores[index], all_ids[index]),
            )[:candidate_count]
            keyword_ids = [all_ids[index] for index in keyword_order]

        rankings = {
            "semantic": [(semantic_ids, 1.0)],
            "keyword": [(keyword_ids, 1.0)],
            "hybrid": [
                (semantic_ids, self.settings.semantic_weight),
                (keyword_ids, self.settings.keyword_weight),
            ],
        }
        fused = reciprocal_rank_fusion(rankings[mode], rrf_k=self.settings.rrf_k)[
            :top_k
        ]

        by_id = {
            item_id: (text, metadata)
            for item_id, text, metadata in zip(all_ids, all_texts, all_metadata, strict=True)
        }
        results: list[RetrievedChunk] = []
        for item_id, score in fused:
            text, metadata = by_id[item_id]
            results.append(
                RetrievedChunk(
                    chunk_id=item_id,
                    text=text,
                    source=str(metadata.get("source", "unknown")),
                    title=str(metadata.get("title") or "") or None,
                    page=int(metadata.get("page") or 0) or None,
                    section=str(metadata.get("section") or "") or None,
                    score=score,
                )
            )
        return results

