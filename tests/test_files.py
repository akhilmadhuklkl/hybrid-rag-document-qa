from app.files import locate_original, remove_original, save_original


def test_original_file_lifecycle(tmp_path):
    path = save_original(tmp_path, "alice", "hash", "../policy.txt", b"policy")
    assert path.name == "policy.txt"
    assert locate_original(tmp_path, "alice", "hash") == path
    assert remove_original(tmp_path, "alice", "hash")
    assert locate_original(tmp_path, "alice", "hash") is None
