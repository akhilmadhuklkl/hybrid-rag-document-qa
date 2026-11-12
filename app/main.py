from contextlib import asynccontextmanager
import hashlib

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from app.config import get_settings
from app.generation import ExtractiveGenerator, OpenAIGenerator
from app.models import (
    Citation,
    DeleteDocumentResponse,
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.parsing import parse_document
from app.store import HybridDocumentStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.store = HybridDocumentStore(settings)
    if settings.generator.lower() == "openai":
        app.state.generator = OpenAIGenerator(settings.openai_model, settings.openai_api_key)
    else:
        app.state.generator = ExtractiveGenerator()
    yield


app = FastAPI(
    title="Hybrid RAG API",
    description="Company-document Q&A using ChromaDB semantic search and BM25.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse, status_code=201)
def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    try:
        count = request.app.state.store.add(payload.documents)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return IngestResponse(documents_received=len(payload.documents), chunks_added=count)


@app.post("/documents/upload", response_model=UploadResponse, status_code=201)
async def upload_documents(
    request: Request, files: list[UploadFile] = File(...)
) -> UploadResponse:
    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    units = []
    duplicates: list[str] = []
    for upload in files:
        filename = upload.filename or "unnamed"
        content = await upload.read(max_bytes + 1)
        await upload.close()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the {settings.max_upload_mb} MB limit",
            )
        content_hash = hashlib.sha256(content).hexdigest()
        if request.app.state.store.has_document(content_hash):
            duplicates.append(filename)
            continue
        try:
            parsed = parse_document(filename, content)
            units.extend(
                unit.model_copy(update={"content_hash": content_hash}) for unit in parsed
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    chunks_added = request.app.state.store.add(units)
    return UploadResponse(
        files_received=len(files),
        document_units_extracted=len(units),
        chunks_added=chunks_added,
        duplicates_skipped=duplicates,
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents(request: Request) -> DocumentListResponse:
    documents = request.app.state.store.list_documents()
    return DocumentListResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=sum(document.chunk_count for document in documents),
    )


@app.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str, request: Request) -> DeleteDocumentResponse:
    deleted = request.app.state.store.delete_document(document_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return DeleteDocumentResponse(document_id=document_id, chunks_deleted=deleted)


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    chunks = request.app.state.store.search(payload.question, payload.top_k)
    answer = request.app.state.generator.generate(payload.question, chunks)
    citations = [
        Citation(
            index=index,
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            title=chunk.title,
            page=chunk.page,
            section=chunk.section,
            text=chunk.text,
            score=round(chunk.score, 8),
        )
        for index, chunk in enumerate(chunks, 1)
    ]
    return QueryResponse(answer=answer, citations=citations)

