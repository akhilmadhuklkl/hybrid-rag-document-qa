from types import SimpleNamespace

from app.main import readiness


class Collection:
    def count(self):
        return 0


class Database:
    def _connect(self):
        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, *_args):
                return self

            def fetchone(self):
                return (1,)

        return Connection()


def test_readiness_reports_dependencies():
    response = readiness(SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        store=SimpleNamespace(collection=Collection()), database=Database()
    ))))
    assert response.status_code == 200
    assert response.body == b'{"status":"ready","checks":{"vector_store":"ok","database":"ok"}}'
