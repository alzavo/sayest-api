import torch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from app.core import lifespan as lifespan_module

    monkeypatch.setattr(
        lifespan_module, "load_model_and_processor", lambda _path: ("model", "proc")
    )

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_predict_happy_path(client, monkeypatch):
    from app.api import routes as routes_module

    monkeypatch.setattr(
        routes_module, "process_audio_bytes", lambda _b: torch.zeros(1, 16000)
    )

    def fake_run_model_inference(_waveform, phonemes, _model, _processor):
        return [1 for _ in phonemes]

    monkeypatch.setattr(routes_module, "run_model_inference", fake_run_model_inference)
    client.app.state.models = {"quality": ("m", "p"), "duration": ("m", "p")}

    response = client.post(
        "/predict",
        data={"phonemes": "a i", "word": "ai"},
        files={"audio": ("test.wav", b"fake", "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["word"] == "ai"
    assert payload["original_transcript"] == "a i"
    assert len(payload["details"]) == 2
    assert payload["details"][0]["phoneme"] == "a"
    assert payload["details"][0]["quality_score"] == 1
    assert payload["details"][0]["duration_score"] == 1


def test_predict_rejects_empty_phonemes(client):
    response = client.post(
        "/predict",
        data={"phonemes": "   ", "word": "ai"},
        files={"audio": ("test.wav", b"fake", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Phonemes string is empty."


def test_predict_rejects_invalid_phoneme(client):
    response = client.post(
        "/predict",
        data={"phonemes": "notaphoneme", "word": "ai"},
        files={"audio": ("test.wav", b"fake", "audio/wav")},
    )

    assert response.status_code == 400
    assert "Invalid phoneme" in response.json()["detail"]


def test_predict_requires_models_loaded(client):
    client.app.state.models = {}

    response = client.post(
        "/predict",
        data={"phonemes": "a i", "word": "ai"},
        files={"audio": ("test.wav", b"fake", "audio/wav")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Models are not loaded."


def test_predict_handles_audio_value_error(client, monkeypatch):
    from app.api import routes as routes_module

    def fake_process_audio_bytes(_b):
        raise ValueError("Audio too long.")

    monkeypatch.setattr(routes_module, "process_audio_bytes", fake_process_audio_bytes)
    client.app.state.models = {"quality": ("m", "p"), "duration": ("m", "p")}

    response = client.post(
        "/predict",
        data={"phonemes": "a i", "word": "ai"},
        files={"audio": ("test.wav", b"fake", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Audio too long."


def test_predict_handles_audio_exception(client, monkeypatch):
    from app.api import routes as routes_module

    def fake_process_audio_bytes(_b):
        raise RuntimeError("Bad audio")

    monkeypatch.setattr(routes_module, "process_audio_bytes", fake_process_audio_bytes)
    client.app.state.models = {"quality": ("m", "p"), "duration": ("m", "p")}

    response = client.post(
        "/predict",
        data={"phonemes": "a i", "word": "ai"},
        files={"audio": ("test.wav", b"fake", "audio/wav")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid audio file."
