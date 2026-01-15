import pytest
from fastapi.testclient import TestClient


def test_lifespan_loads_and_unloads_models(monkeypatch):
    from app.core import lifespan as lifespan_module

    monkeypatch.setattr(
        lifespan_module, "load_model_and_processor", lambda _path: ("model", "proc")
    )

    from app.main import app

    with TestClient(app) as client:
        assert "quality" in client.app.state.models
        assert "duration" in client.app.state.models

    assert not app.state.models


def test_lifespan_logs_and_aborts_on_load_failure(monkeypatch):
    from app.core import lifespan as lifespan_module

    def fail_load(_path):
        raise RuntimeError("boom")

    monkeypatch.setattr(lifespan_module, "load_model_and_processor", fail_load)

    from app.main import app

    with pytest.raises(RuntimeError, match="generator didn't yield"):
        with TestClient(app):
            pass

    assert not hasattr(app.state, "models") or not app.state.models
