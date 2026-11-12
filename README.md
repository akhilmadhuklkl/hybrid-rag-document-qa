# Hybrid RAG Pipeline

A small, production-shaped retrieval-augmented generation service that combines:

- semantic search with ChromaDB embeddings;
- keyword search with BM25;
- reciprocal-rank fusion (RRF) to merge both rankings;
- source-aware answers with inline citations;
- PDF page and DOCX section-aware citations;
- a FastAPI interface for ingestion and querying.

The service works without an LLM: it returns an extractive answer assembled from the best passages. If `OPENAI_API_KEY` is set, it can instead synthesize an answer with an OpenAI model while being constrained to the retrieved context.

## Quick start

Python 3.11 or 3.12 is recommended (some dependencies may not yet support Python 3.14).

```powershell
cd project-1-hybrid-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Using `python -m uvicorn` also works on managed Windows machines where Application
Control blocks the generated `uvicorn.exe` launcher.

Open `http://127.0.0.1:8001/docs` for the interactive API documentation. If port
`8001` is unavailable on another machine, select a different unreserved local port.

## Use the API

Ingest one or more documents:

```powershell
$body = @{
  documents = @(
    @{
      source = "employee-handbook.md"
      title = "Employee Handbook"
      text = "Employees receive 20 days of annual leave. Requests require manager approval."
    }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/documents -ContentType application/json -Body $body
```

Ask a question:

```powershell
$body = @{ question = "How much annual leave do employees receive?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/query -ContentType application/json -Body $body
```

Every answer includes a `citations` array containing the source, title, chunk ID, passage text, and fused relevance score. Inline markers such as `[1]` correspond to that array.

### Upload files directly

The `/documents/upload` endpoint accepts PDF, DOCX, UTF-8 TXT, Markdown, or multiple
files in one request. PDF pages and DOCX headings are retained in query citations.

```powershell
curl.exe -X POST "http://127.0.0.1:8001/documents/upload" `
  -F "files=@C:\docs\employee-handbook.pdf" `
  -F "files=@C:\docs\benefits.docx"
```

Scanned image-only PDFs need OCR before upload. Each file is limited to 20 MB by
default; change `MAX_UPLOAD_MB` to adjust the limit.

Uploading identical file contents again is safe: the API detects its SHA-256 hash,
does not create duplicate chunks, and returns the filename in `duplicates_skipped`.

### Manage indexed documents

List documents and collection totals:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/documents
```

Each item includes a stable `document_id`, source, title, chunk count, page count,
sections, and ingestion timestamp. Delete a document and all of its chunks using:

```powershell
Invoke-RestMethod -Method Delete `
  -Uri "http://127.0.0.1:8001/documents/{document_id}"
```

Documents indexed before this feature are exposed with a `legacy:` identifier and
remain manageable through the same endpoint.

## Optional OpenAI generation

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:RAG_GENERATOR = "openai"
$env:OPENAI_MODEL = "gpt-4.1-mini"
```

Configuration is read from environment variables. Useful settings include `CHROMA_PATH`, `COLLECTION_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `SEMANTIC_WEIGHT`, and `KEYWORD_WEIGHT`.

## Test

```powershell
pytest
```

## Evaluate retrieval quality

The included benchmark compares semantic-only, BM25-only, and hybrid retrieval
against labeled source documents. It reports Recall@K, mean reciprocal rank (MRR),
and latency without touching the production Chroma collection.

```powershell
python -m evaluation.run --top-k 3
```

Save a machine-readable result for experiment tracking:

```powershell
python -m evaluation.run --top-k 3 --json-output evaluation/results/baseline.json
```

Extend `evaluation/sample_dataset.json` with real representative questions and
their expected sources before tuning chunk sizes, retrieval weights, or reranking.

## Architecture

1. Documents are split into overlapping word chunks.
2. Chunks and metadata are persisted in ChromaDB.
3. At query time, Chroma produces a semantic ranking and BM25 produces a lexical ranking.
4. Weighted RRF merges the rankings without requiring their raw scores to share a scale.
5. The top passages are passed to the selected answer generator and returned as traceable citations.

