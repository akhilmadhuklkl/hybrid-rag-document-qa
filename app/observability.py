from collections import Counter
import json
import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                "level": record.levelname,
                "message": record.getMessage(),
            },
            separators=(",", ":"),
        )


class RequestMetrics:
    def __init__(self):
        self.requests = 0
        self.errors = 0
        self.total_duration_ms = 0.0
        self.routes: Counter[str] = Counter()

    def snapshot(self) -> dict[str, object]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "average_latency_ms": round(self.total_duration_ms / self.requests, 2)
            if self.requests
            else 0.0,
            "routes": dict(self.routes),
        }


class ObservabilityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, metrics: RequestMetrics):
        super().__init__(app)
        self.metrics = metrics
        self.logger = logging.getLogger("atlas.api")

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            self.metrics.errors += 1
            raise
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            self.metrics.requests += 1
            self.metrics.total_duration_ms += duration_ms
            self.metrics.routes[request.url.path] += 1
            self.logger.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": status_code,
                        "duration_ms": round(duration_ms, 2),
                    },
                    separators=(",", ":"),
                )
            )
        response.headers["X-Request-ID"] = request_id
        return response
