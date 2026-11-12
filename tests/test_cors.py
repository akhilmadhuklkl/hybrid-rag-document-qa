from app.main import app


def test_cors_allows_conversation_rename_method():
    middleware = next(item for item in app.user_middleware if item.cls.__name__ == "CORSMiddleware")
    assert "PATCH" in middleware.kwargs["allow_methods"]
    assert "OPTIONS" in middleware.kwargs["allow_methods"]
