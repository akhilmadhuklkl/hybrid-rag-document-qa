from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from typing import Any, Literal

import chromadb
from rank_bm25 import BM25Okapi

from app.config import Settings
from app.models import DocumentInput, DocumentSummary
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
    evidence_score: float = 0.0


class HybridDocumentStore:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        self.client = client or chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, documents: list[DocumentInput], owner_id: str = "demo-user") -> int:
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ingested_at = datetime.now(UTC).isoformat()
        for document in documents:
            document_hash = document.content_hash or hashlib.sha256(
                f"{document.source}\0{document.text}".encode()
            ).hexdigest()
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
                        "document_hash": document_hash,
                        "owner_id": document.owner_id or owner_id,
                        "ingested_at": ingested_at,
                    }
                )
        if ids:
            self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        return len(ids)

    def has_document(self, document_hash: str, owner_id: str = "demo-user") -> bool:
        result = self.collection.get(
            where={"$and": [{"document_hash": document_hash}, {"owner_id": owner_id}]}, limit=1, include=[]
        )
        return bool(result.get("ids"))

    def list_documents(self, owner_id: str = "demo-user") -> list[DocumentSummary]:
        stored = self.collection.get(where={"owner_id": owner_id}, include=["metadatas"])
        ids = stored.get("ids", [])
        metadatas = stored.get("metadatas") or []
        grouped: dict[str, dict[str, Any]] = {}
        for chunk_id_value, metadata in zip(ids, metadatas, strict=True):
            source = str(metadata.get("source", "unknown"))
            document_id = str(metadata.get("document_hash") or f"legacy:{source}")
            item = grouped.setdefault(
                document_id,
                {
                    "source": source,
                    "title": str(metadata.get("title") or "") or None,
                    "chunk_ids": [],
                    "pages": set(),
                    "sections": set(),
                    "ingested_at": str(metadata.get("ingested_at") or "") or None,
                },
            )
            item["chunk_ids"].append(chunk_id_value)
            if page := int(metadata.get("page") or 0):
                item["pages"].add(page)
            if section := str(metadata.get("section") or ""):
                item["sections"].add(section)

        return sorted(
            [
                DocumentSummary(
                    document_id=document_id,
                    source=item["source"],
                    title=item["title"],
                    chunk_count=len(item["chunk_ids"]),
                    page_count=len(item["pages"]),
                    sections=sorted(item["sections"]),
                    ingested_at=item["ingested_at"],
                )
                for document_id, item in grouped.items()
            ],
            key=lambda item: (item.source.lower(), item.document_id),
        )

    def delete_document(self, document_id: str, owner_id: str = "demo-user") -> int:
        if document_id.startswith("legacy:"):
            source = document_id.removeprefix("legacy:")
            matched = self.collection.get(where={"$and": [{"source": source}, {"owner_id": owner_id}]}, include=[])
        else:
            matched = self.collection.get(
                where={"$and": [{"document_hash": document_id}, {"owner_id": owner_id}]}, include=[]
            )
        ids = matched.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def search(
        self,
        question: str,
        top_k: int,
        mode: Literal["semantic", "keyword", "hybrid"] = "hybrid",
        owner_id: str | None = "demo-user",
    ) -> list[RetrievedChunk]:
        where = {"owner_id": owner_id} if owner_id else None
        stored = self.collection.get(where=where, include=["documents", "metadatas"])
        all_ids = stored.get("ids", [])
        all_texts = stored.get("documents") or []
        all_metadata = stored.get("metadatas") or []
        if not all_ids:
            return []

        candidate_count = min(len(all_ids), max(top_k * 4, top_k))
        semantic_ids: list[str] = []
        semantic_scores: dict[str, float] = {}
        keyword_ids: list[str] = []
        if mode in {"semantic", "hybrid"}:
            semantic = self.collection.query(
                query_texts=[question],
                n_results=candidate_count,
                include=["documents", "distances"],
                where=where,
            )
            semantic_ids = (semantic.get("ids") or [[]])[0]
            semantic_distances = (semantic.get("distances") or [[]])[0]
            semantic_scores = {
                item_id: max(0.0, min(1.0, 1.0 - float(distance)))
                for item_id, distance in zip(
                    semantic_ids, semantic_distances, strict=True
                )
            }

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
                    evidence_score=semantic_scores.get(item_id, 0.0),
                )
            )
        return results

