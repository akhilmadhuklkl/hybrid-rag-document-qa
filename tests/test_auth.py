import os

import pytest
from fastapi import HTTPException

from app.auth import current_user
from app.config import get_settings


def request_with_headers(headers):
    return type("Request", (), {"headers": headers})()


def test_auth_disabled_uses_user_header(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    get_settings.cache_clear()
    assert current_user(request_with_headers({"x-user-id": "alice"})) == "alice"


def test_auth_enabled_requires_matching_api_key(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY", "secret")
    get_settings.cache_clear()
    with pytest.raises(HTTPException) as error:
        current_user(request_with_headers({}))
    assert error.value.status_code == 401
    assert current_user(request_with_headers({"x-api-key": "secret", "x-user-id": "alice"})) == "alice"
    monkeypatch.setenv("AUTH_ENABLED", "false")
    get_settings.cache_clear()
