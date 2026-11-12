from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_ui_exposes_document_source_filter():
    html = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'id="document-filter"' in html
    assert "source=${encodeURIComponent(filter)}" in javascript
