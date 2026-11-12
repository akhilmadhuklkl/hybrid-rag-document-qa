from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    source: str = Field(min_length=1, examples=["employee-handbook.md"])
    text: str = Field(min_length=1)
    title: str | None = None
    page: int | None = Field(default=None, ge=1)
    section: str | None = None


class IngestRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1)


class IngestResponse(BaseModel):
    documents_received: int
    chunks_added: int


class UploadResponse(BaseModel):
    files_received: int
    document_units_extracted: int
    chunks_added: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    index: int
    chunk_id: str
    source: str
    title: str | None = None
    page: int | None = None
    section: str | None = None
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]

