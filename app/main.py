from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.generation import ExtractiveGenerator, OpenAIGenerator, has_valid_citations
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
from app.reranking import CrossEncoderReranker
from app.store import HybridDocumentStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.store = HybridDocumentStore(settings)
    app.state.reranker = (
        CrossEncoderReranker(settings.reranker_model)
        if settings.reranker_enabled
        else None
    )
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

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
    return build_query_response(payload, request)


def build_query_response(payload: QueryRequest, request: Request) -> QueryResponse:
    reranker = request.app.state.reranker
    candidate_k = (
        payload.top_k * get_settings().reranker_candidate_multiplier
        if reranker
        else payload.top_k
    )
    chunks = request.app.state.store.search(payload.question, candidate_k)
    if reranker:
        chunks = reranker.rerank(payload.question, chunks, payload.top_k)
    chunks = [
        chunk
        for chunk in chunks
        if chunk.evidence_score >= get_settings().min_evidence_score
    ][: payload.top_k]
    generator = request.app.state.generator
    answer = generator.generate(payload.question, chunks)
    generation_mode = "openai" if isinstance(generator, OpenAIGenerator) else "extractive"
    grounded = not chunks or has_valid_citations(answer, len(chunks))
    if chunks and not grounded:
        answer = ExtractiveGenerator().generate(payload.question, chunks)
        generation_mode = "extractive_fallback"
        grounded = True
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
            evidence_score=round(chunk.evidence_score, 8),
        )
        for index, chunk in enumerate(chunks, 1)
    ]
    return QueryResponse(
        answer=answer,
        citations=citations,
        reranked=reranker is not None,
        grounded=grounded,
        generation_mode=generation_mode,
    )


@app.post("/query/stream")
def stream_query(payload: QueryRequest, request: Request) -> StreamingResponse:
    response = build_query_response(payload, request)

    def events():
        for word in response.answer.split():
            yield f"event: token\ndata: {json.dumps({'text': word + ' '})}\n\n"
        metadata = response.model_dump(exclude={"answer"})
        yield f"event: complete\ndata: {json.dumps(metadata)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")

