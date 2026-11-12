from app.database import ConversationStore
from app.jobs import IngestionJobManager
from app.models import DocumentInput


class FakeStore:
    def add(self, documents):
        return len(documents) * 2


def test_job_manager_completes_background_work(tmp_path):
    database = ConversationStore(tmp_path / "atlas.db")
    manager = IngestionJobManager(FakeStore(), database, workers=1)
    job = manager.submit([DocumentInput(source="a.txt", text="hello")], 1)
    for _ in range(30):
        current = manager.get(job.job_id)
        if current.status in {"completed", "failed"}:
            break
    assert current.status == "completed"
    assert current.chunks_added == 2
    manager.shutdown()
