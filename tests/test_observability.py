from app.observability import RequestMetrics


def test_metrics_snapshot_is_safe_before_any_requests():
    metrics = RequestMetrics()
    assert metrics.snapshot() == {
        "requests": 0,
        "errors": 0,
        "average_latency_ms": 0.0,
        "routes": {},
    }
