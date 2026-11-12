from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from app.config import get_settings
from app.generation import ExtractiveGenerator, OpenAIGenerator
from app.models import (
    Citation,
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
    for upload in files:
        filename = upload.filename or "unnamed"
        content = await upload.read(max_bytes + 1)
        await upload.close()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the {settings.max_upload_mb} MB limit",
            )
        try:
            units.extend(parse_document(filename, content))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    chunks_added = request.app.state.store.add(units)
    return UploadResponse(
        files_received=len(files),
        document_units_extracted=len(units),
        chunks_added=chunks_added,
    )


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

