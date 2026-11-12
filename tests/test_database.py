from app.database import ConversationStore
from app.models import Citation


def test_conversation_store_persists_exchange(tmp_path):
    store = ConversationStore(tmp_path / "atlas.db")
    citation = Citation(index=1, chunk_id="abc", source="policy.md", text="Answer", score=0.4, evidence_score=0.8)
    conversation_id = store.save_exchange(None, "How many days?", "Twenty days [1].", [citation])
    summaries = store.list_conversations()
    assert summaries[0].conversation_id == conversation_id
    detail = store.get_conversation(conversation_id)
    assert detail is not None
    assert [message.role for message in detail.messages] == ["user", "assistant"]
    assert detail.messages[1].citations[0].source == "policy.md"
