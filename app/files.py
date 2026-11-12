from pathlib import Path


def save_original(root: Path, owner_id: str, document_id: str, filename: str, content: bytes) -> Path:
    safe_name = Path(filename).name
    destination = root / owner_id / document_id
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / safe_name
    path.write_bytes(content)
    return path


def locate_original(root: Path, owner_id: str, document_id: str) -> Path | None:
    directory = root / owner_id / document_id
    if not directory.is_dir():
        return None
    files = [path for path in directory.iterdir() if path.is_file()]
    return files[0] if files else None


def remove_original(root: Path, owner_id: str, document_id: str) -> bool:
    directory = root / owner_id / document_id
    if not directory.is_dir():
        return False
    removed = False
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()
            removed = True
    directory.rmdir()
    return removed
