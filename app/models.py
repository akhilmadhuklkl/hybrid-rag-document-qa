from typing import Literal

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    source: str = Field(min_length=1, examples=["employee-handbook.md"])
    text: str = Field(min_length=1)
    title: str | None = None
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    content_hash: str | None = Field(default=None, exclude=True)
    owner_id: str | None = Field(default=None, exclude=True)


class IngestRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1)


class IngestResponse(BaseModel):
    documents_received: int
    chunks_added: int


class UploadResponse(BaseModel):
    files_received: int
    document_units_extracted: int
    chunks_added: int
    duplicates_skipped: list[str] = Field(default_factory=list)


class IngestionJob(BaseModel):
    job_id: str
    status: str
    files_received: int
    document_units: int
    chunks_added: int = 0
    error: str | None = None
    created_at: str
    updated_at: str


class UploadJobResponse(BaseModel):
    jobs: list[IngestionJob]
    duplicates_skipped: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    document_id: str
    source: str
    title: str | None = None
    chunk_count: int
    page_count: int
    sections: list[str]
    ingested_at: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total_documents: int
    total_chunks: int


class DeleteDocumentResponse(BaseModel):
    document_id: str
    chunks_deleted: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)
    conversation_id: str | None = None
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"


class Citation(BaseModel):
    index: int
    chunk_id: str
    document_id: str | None = None
    source: str
    title: str | None = None
    page: int | None = None
    section: str | None = None
    text: str
    score: float
    evidence_score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    reranked: bool = False
    grounded: bool = True
    generation_mode: str = "extractive"
    conversation_id: str
    retrieval_mode: str = "hybrid"
    retrieved_chunks: int = 0
    latency_ms: float = 0.0


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationMessage(BaseModel):
    role: str
    content: str
    citations: list[Citation] = Field(default_factory=list)
    created_at: str


class ConversationDetail(ConversationSummary):
    messages: list[ConversationMessage]


class ConversationRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)


class ConversationDeleteResponse(BaseModel):
    conversation_id: str
    deleted: bool

