from contextlib import asynccontextmanager
import hashlib
import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.auth import current_user
from app.database import ConversationStore
from app.generation import ExtractiveGenerator, OpenAIGenerator, has_valid_citations
from app.jobs import IngestionJobManager
from app.files import locate_original, remove_original, save_original
from app.observability import JsonFormatter, ObservabilityMiddleware, RequestMetrics
from app.security import SecurityMiddleware
from app.models import (
    Citation,
    DeleteDocumentResponse,
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
    ConversationSummary,
    ConversationDetail,
    ConversationRenameRequest,
    ConversationDeleteResponse,
    UploadJobResponse,
    IngestionJob,
)
from app.parsing import parse_document
from app.reranking import CrossEncoderReranker
from app.store import HybridDocumentStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.store = HybridDocumentStore(settings)
    app.state.database = ConversationStore(settings.database_path)
    app.state.jobs = IngestionJobManager(app.state.store, app.state.database)
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
    app.state.jobs.shutdown()


app = FastAPI(
    title="Hybrid RAG API",
    description="Company-document Q&A using ChromaDB semantic search and BM25.",
    version="1.0.0",
    lifespan=lifespan,
)

metrics = RequestMetrics()
settings = get_settings()
app.add_middleware(SecurityMiddleware, limit_per_minute=settings.rate_limit_per_minute, max_request_mb=settings.max_request_mb)
app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()], allow_credentials=True, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
app.add_middleware(ObservabilityMiddleware, metrics=metrics)
logging_handler = logging.StreamHandler()
logging_handler.setFormatter(JsonFormatter())
logging.getLogger("atlas.api").addHandler(logging_handler)
logging.getLogger("atlas.api").setLevel(logging.INFO)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics_endpoint() -> dict[str, object]:
    return metrics.snapshot()


@app.post("/documents", response_model=IngestResponse, status_code=201)
def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    owner_id = current_user(request)
    try:
        documents = [document.model_copy(update={"owner_id": owner_id}) for document in payload.documents]
        count = request.app.state.store.add(documents, owner_id=owner_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return IngestResponse(documents_received=len(payload.documents), chunks_added=count)


@app.post("/documents/upload", response_model=UploadResponse, status_code=201)
async def upload_documents(
    request: Request, files: list[UploadFile] = File(...)
) -> UploadResponse:
    settings = get_settings()
    owner_id = current_user(request)
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
        if request.app.state.store.has_document(content_hash, owner_id):
            duplicates.append(filename)
            continue
        try:
            parsed = parse_document(filename, content)
            save_original(settings.file_storage_path, owner_id, content_hash, filename, content)
            units.extend(unit.model_copy(update={"content_hash": content_hash, "owner_id": owner_id}) for unit in parsed)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    chunks_added = request.app.state.store.add(units)
    return UploadResponse(
        files_received=len(files),
        document_units_extracted=len(units),
        chunks_added=chunks_added,
        duplicates_skipped=duplicates,
    )


@app.post("/documents/upload/async", response_model=UploadJobResponse, status_code=202)
async def upload_documents_async(
    request: Request, files: list[UploadFile] = File(...)
) -> UploadJobResponse:
    settings = get_settings()
    owner_id = current_user(request)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    units = []
    duplicates: list[str] = []
    for upload in files:
        filename = upload.filename or "unnamed"
        content = await upload.read(max_bytes + 1)
        await upload.close()
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail=f"{filename} exceeds the {settings.max_upload_mb} MB limit")
        content_hash = hashlib.sha256(content).hexdigest()
        if request.app.state.store.has_document(content_hash, owner_id):
            duplicates.append(filename)
            continue
        try:
            parsed = parse_document(filename, content)
            save_original(settings.file_storage_path, owner_id, content_hash, filename, content)
            units.extend(unit.model_copy(update={"content_hash": content_hash, "owner_id": owner_id}) for unit in parsed)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not units:
        return UploadJobResponse(jobs=[], duplicates_skipped=duplicates)
    job = request.app.state.jobs.submit(units, len(files))
    return UploadJobResponse(jobs=[job], duplicates_skipped=duplicates)


@app.get("/jobs/{job_id}", response_model=IngestionJob)
def get_ingestion_job(job_id: str, request: Request) -> IngestionJob:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job


@app.get("/documents", response_model=DocumentListResponse)
def list_documents(request: Request) -> DocumentListResponse:
    documents = request.app.state.store.list_documents(current_user(request))
    return DocumentListResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=sum(document.chunk_count for document in documents),
    )


@app.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str, request: Request) -> DeleteDocumentResponse:
    owner_id = current_user(request)
    deleted = request.app.state.store.delete_document(document_id, owner_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    remove_original(get_settings().file_storage_path, owner_id, document_id)
    return DeleteDocumentResponse(document_id=document_id, chunks_deleted=deleted)


@app.get("/documents/{document_id}/file", include_in_schema=False)
def source_file(document_id: str, request: Request) -> FileResponse:
    path = locate_original(get_settings().file_storage_path, current_user(request), document_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Original source file not found")
    return FileResponse(path)


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    return build_query_response(payload, request)


def build_query_response(payload: QueryRequest, request: Request) -> QueryResponse:
    owner_id = current_user(request)
    reranker = request.app.state.reranker
    candidate_k = (
        payload.top_k * get_settings().reranker_candidate_multiplier
        if reranker
        else payload.top_k
    )
    chunks = request.app.state.store.search(payload.question, candidate_k, owner_id=owner_id)
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
            document_id=chunk.document_id,
            page=chunk.page,
            section=chunk.section,
            text=chunk.text,
            score=round(chunk.score, 8),
            evidence_score=round(chunk.evidence_score, 8),
        )
        for index, chunk in enumerate(chunks, 1)
    ]
    conversation_id = request.app.state.database.save_exchange(
        payload.conversation_id, payload.question, answer, citations, owner_id
    )
    return QueryResponse(
        answer=answer,
        citations=citations,
        reranked=reranker is not None,
        grounded=grounded,
        generation_mode=generation_mode,
        conversation_id=conversation_id,
    )


@app.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(request: Request) -> list[ConversationSummary]:
    return request.app.state.database.list_conversations(current_user(request))


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, request: Request) -> ConversationDetail:
    conversation = request.app.state.database.get_conversation(conversation_id, current_user(request))
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
def rename_conversation(conversation_id: str, payload: ConversationRenameRequest, request: Request) -> ConversationSummary:
    owner_id = current_user(request)
    if not request.app.state.database.rename_conversation(conversation_id, payload.title, owner_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    conversations = request.app.state.database.list_conversations(owner_id)
    return next(item for item in conversations if item.conversation_id == conversation_id)


@app.delete("/conversations/{conversation_id}", response_model=ConversationDeleteResponse)
def delete_conversation(conversation_id: str, request: Request) -> ConversationDeleteResponse:
    if not request.app.state.database.delete_conversation(conversation_id, current_user(request)):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDeleteResponse(conversation_id=conversation_id, deleted=True)


@app.post("/query/stream")
def stream_query(payload: QueryRequest, request: Request) -> StreamingResponse:
    response = build_query_response(payload, request)

    def events():
        for word in response.answer.split():
            yield f"event: token\ndata: {json.dumps({'text': word + ' '})}\n\n"
        metadata = response.model_dump(exclude={"answer"})
        yield f"event: complete\ndata: {json.dumps(metadata)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")

