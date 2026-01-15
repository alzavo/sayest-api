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


def test_process_audio_bytes_no_resample_when_sampling_rate_matches(monkeypatch):
    waveform = torch.ones((1, 16000))

    def fake_load(_fileobj):
        return waveform, SAMPLING_RATE

    def fail_resample(*_args, **_kwargs):
        raise AssertionError("Resample should not be called")

    monkeypatch.setattr(utils_module.torchaudio, "load", fake_load)
    monkeypatch.setattr(utils_module.torchaudio.transforms, "Resample", fail_resample)

    result = utils_module.process_audio_bytes(b"fake")

    assert result.shape == waveform.shape


def test_run_model_inference_scores(monkeypatch):
    class FakeTokenizer:
        unk_token_id = 7

        def __init__(self):
            self.seen_tokens = None

        def convert_tokens_to_ids(self, tokens):
            self.seen_tokens = tokens
            return [5, None]

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

        def __call__(self, audio_input, sampling_rate, return_tensors, padding):
            assert sampling_rate == SAMPLING_RATE
            assert return_tensors == "pt"
            assert padding is True
            input_values = audio_input.unsqueeze(0)
            attention_mask = torch.ones_like(input_values, dtype=torch.long)
            return type(
                "Processed", (), {"input_values": input_values, "attention_mask": attention_mask}
            )

    class FakeOutputs:
        def __init__(self, logits):
            self.logits = logits

    class FakeModel:
        device = torch.device("cpu")

        def __call__(
            self,
            input_values,
            attention_mask,
            canonical_token_ids,
            token_lengths,
            token_mask,
        ):
            # Ensure unknown tokens are replaced with unk_token_id before inference.
            assert canonical_token_ids[0, 0].item() == 5
            assert canonical_token_ids[0, 1].item() == 7
            assert input_values.shape[0] == 1
            assert attention_mask.shape == input_values.shape
            assert canonical_token_ids.shape[1] == 2
            assert token_lengths.item() == 2
            assert token_mask.shape == canonical_token_ids.shape
            # Argmax index per phoneme (0-based) is then shifted by +1 to match 1..N scoring.
            scores = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 1.0, 3.0]]])
            return FakeOutputs({"head": scores})

    waveform = torch.zeros(1, 16000)
    processor = FakeProcessor()
    model = FakeModel()

    scores = utils_module.run_model_inference(waveform, ["a", "i"], model, processor)

    assert processor.tokenizer.seen_tokens == ["a", "i"]
    assert scores == [1, 3]
