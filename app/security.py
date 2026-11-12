from collections import defaultdict
from threading import Lock
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


def parse_content_length(value: str | None) -> int:
    if not value:
        return 0
    try:
        length = int(value)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length header") from exc
    if length < 0:
        raise ValueError("Invalid Content-Length header")
    return length


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_minute: int, max_request_mb: int):
        super().__init__(app)
        self.limit = limit_per_minute
        self.max_bytes = max_request_mb * 1024 * 1024
        self._lock = Lock()
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        try:
            declared_length = parse_content_length(content_length)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        if declared_length > self.max_bytes:
            return JSONResponse(status_code=413, content={"detail": "Request body exceeds configured limit"})
        client = request.client.host if request.client else "unknown"
        now = time.time()
        with self._lock:
            recent = [stamp for stamp in self._requests[client] if now - stamp < 60]
            if len(recent) >= self.limit:
                return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"}, headers={"Retry-After": "60"})
            recent.append(now)
            self._requests[client] = recent
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        return response
