from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_ui_exposes_retrieval_mode_selector_and_sends_it():
    html = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'id="retrieval-mode"' in html
    assert 'value="semantic"' in html
    assert "mode: elements.retrievalMode.value" in javascript
    assert 'elements.form.setAttribute("aria-busy", "true")' in javascript
    assert 'elements.retrievalMode.disabled = true' in javascript
    assert "retrieved_chunks" in javascript
