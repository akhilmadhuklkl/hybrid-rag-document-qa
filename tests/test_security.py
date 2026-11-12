from app.security import SecurityMiddleware


def test_security_middleware_has_configured_limits():
    middleware = SecurityMiddleware(lambda scope, receive, send: None, 10, 3)
    assert middleware.limit == 10
    assert middleware.max_bytes == 3 * 1024 * 1024
