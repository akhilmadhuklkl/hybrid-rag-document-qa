import pytest

from app.security import SecurityMiddleware, parse_content_length


def test_security_middleware_has_configured_limits():
    middleware = SecurityMiddleware(lambda scope, receive, send: None, 10, 3)
    assert middleware.limit == 10
    assert middleware.max_bytes == 3 * 1024 * 1024


def test_content_length_parser_rejects_invalid_values():
    assert parse_content_length(None) == 0
    assert parse_content_length("42") == 42
    with pytest.raises(ValueError, match="Invalid Content-Length"):
        parse_content_length("nope")
    with pytest.raises(ValueError, match="Invalid Content-Length"):
        parse_content_length("-1")
