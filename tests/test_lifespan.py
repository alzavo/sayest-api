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
