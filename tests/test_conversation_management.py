from app.database import ConversationStore


def test_conversation_can_be_renamed_and_deleted(tmp_path):
    store = ConversationStore(tmp_path / "atlas.db")
    conversation_id = store.save_exchange(None, "Question", "Answer", [])
    assert store.rename_conversation(conversation_id, "Renamed", "demo-user")
    assert store.get_conversation(conversation_id).title == "Renamed"
    assert store.delete_conversation(conversation_id, "demo-user")
    assert store.get_conversation(conversation_id) is None
