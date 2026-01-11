import torch
import pytest

from app.constants.audio_properties import MAX_AUDIO_DURATION_SECONDS, SAMPLING_RATE
from app.model import utils as utils_module


def test_process_audio_bytes_resamples_and_mono(monkeypatch):
    waveform = torch.ones((2, 8000))

    def fake_load(_fileobj):
        return waveform, 8000

    resample_called = {"value": False}

    class FakeResample:
        def __init__(self, orig_freq, new_freq):
            self.orig_freq = orig_freq
            self.new_freq = new_freq

        def __call__(self, data):
            resample_called["value"] = True
            return data

    monkeypatch.setattr(utils_module.torchaudio, "load", fake_load)
    monkeypatch.setattr(utils_module.torchaudio.transforms, "Resample", FakeResample)

    result = utils_module.process_audio_bytes(b"fake")

    assert result.shape[0] == 1
    assert resample_called["value"] is True


def test_process_audio_bytes_rejects_too_long(monkeypatch):
    num_samples = int(MAX_AUDIO_DURATION_SECONDS * SAMPLING_RATE) + 1
    waveform = torch.zeros((1, num_samples))

    def fake_load(_fileobj):
        return waveform, SAMPLING_RATE

    monkeypatch.setattr(utils_module.torchaudio, "load", fake_load)

    with pytest.raises(ValueError) as exc:
        utils_module.process_audio_bytes(b"fake")

    assert "Audio too long" in str(exc.value)
