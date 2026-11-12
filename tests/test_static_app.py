from pathlib import Path


STATIC_DIR = Path(__file__).parents[1] / "app" / "static"


def test_web_app_assets_exist_and_reference_api_routes():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'id="query-form"' in html
    assert 'id="drop-zone"' in html
    assert '"/query/stream"' in javascript
    assert '"/documents/upload/async"' in javascript
    assert "deleteDocument" in javascript
