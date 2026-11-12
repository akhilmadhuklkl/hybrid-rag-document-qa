from concurrent.futures import ThreadPoolExecutor
import uuid

from app.database import ConversationStore
from app.models import DocumentInput, IngestionJob
from app.store import HybridDocumentStore


class IngestionJobManager:
    def __init__(self, store: HybridDocumentStore, database: ConversationStore, workers: int = 2):
        self.store = store
        self.database = database
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="atlas-ingest")

    def submit(self, documents: list[DocumentInput], files_received: int) -> IngestionJob:
        job_id = uuid.uuid4().hex
        job = self.database.create_job(job_id, files_received, len(documents))
        self.executor.submit(self._run, job_id, documents)
        return job

    def _run(self, job_id: str, documents: list[DocumentInput]) -> None:
        self.database.update_job(job_id, "processing")
        try:
            chunks_added = self.store.add(documents)
            self.database.update_job(job_id, "completed", chunks_added=chunks_added)
        except Exception as exc:
            self.database.update_job(job_id, "failed", error=str(exc))

    def get(self, job_id: str) -> IngestionJob | None:
        return self.database.get_job(job_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
