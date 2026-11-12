# Portfolio copy

## Resume bullet

Built Atlas, a production-shaped hybrid RAG document QA platform using FastAPI,
ChromaDB, BM25, reciprocal-rank fusion, optional cross-encoder reranking, and
grounded citation generation; added async ingestion, user isolation,
conversation history, observability, security middleware, Docker, CI coverage
gates, and 43 automated tests.

## Short project description

Atlas answers questions over company documents by combining semantic and lexical
retrieval, then returns evidence-linked answers with transparent citations. It
supports PDF, DOCX, Markdown, and text ingestion, configurable retrieval modes,
streaming responses, conversation history, and local Docker deployment.

## Demo video script (60–90 seconds)

1. Open the Atlas web UI and show the document library.
2. Upload `examples/demo_documents.json` through the API or upload a Markdown file.
3. Ask: “How many annual leave days do employees receive?”
4. Show the grounded answer and citation card linking to the source document.
5. Switch Search mode from Hybrid to Keyword and run the same question.
6. Filter the document library by source name.
7. Open `/docs` to show the FastAPI schema.
8. End with the architecture diagram and GitHub release page.

## Screenshot placement

Save the final browser capture as `docs/screenshots/atlas-ui.png`, then add this
line to the README:

```markdown
![Atlas UI](docs/screenshots/atlas-ui.png)
```
