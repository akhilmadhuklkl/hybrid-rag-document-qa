"""Run a minimal health, ingestion, and citation smoke test against a deployment."""

import argparse
import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request(base_url: str, path: str, method: str = "GET", payload: dict | None = None, api_key: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    body = json.dumps(payload).encode() if payload is not None else None
    with urlopen(Request(f"{base_url.rstrip('/')}{path}", data=body, headers=headers, method=method), timeout=30) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("--api-key")
    args = parser.parse_args()
    try:
        health = request(args.base_url, "/health")
        ready = request(args.base_url, "/ready")
        ingest = request(
            args.base_url,
            "/documents",
            "POST",
            {"documents": [{"source": "deployment-smoke.md", "text": "The deployment smoke test policy allows twenty leave days."}]},
            args.api_key,
        )
        query = request(
            args.base_url,
            "/query",
            "POST",
            {"question": "How many leave days are allowed?", "mode": "hybrid"},
            args.api_key,
        )
    except HTTPError as error:
        print(f"Smoke test failed: HTTP {error.code} {error.reason}", file=sys.stderr)
        return 1
    print(json.dumps({
        "health": health.get("status"),
        "ready": ready.get("status"),
        "chunks_added": ingest.get("chunks_added"),
        "citations": len(query.get("citations", [])),
        "grounded": query.get("grounded"),
    }, indent=2))
    return 0 if health.get("status") == "ok" and ready.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
