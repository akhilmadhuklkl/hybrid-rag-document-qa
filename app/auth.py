import secrets

from fastapi import HTTPException, Request

from app.config import get_settings


def current_user(request: Request) -> str:
    settings = get_settings()
    if settings.auth_enabled:
        supplied = request.headers.get("x-api-key", "")
        if not settings.api_key or not secrets.compare_digest(supplied, settings.api_key):
            raise HTTPException(status_code=401, detail="Valid X-API-Key required")
    return request.headers.get("x-user-id", "demo-user")
