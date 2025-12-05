import torch
import torchaudio
from io import BytesIO
from transformers import Wav2Vec2Processor, QuantoConfig
from app.model.gop_model import GOPPhonemeClassifier
from app.constants.audio_properties import (
    MAX_AUDIO_DURATION_SECONDS,
    MONO_CHANNEL,
    SAMPLING_RATE,
)
import logging

logger = logging.getLogger(__name__)


def load_model_and_processor(model_repo_id: str):
    """Loads a specific model and processor."""
    logger.info(f"Loading model: {model_repo_id}")
    quantization_config = QuantoConfig(weights="int8")

    model = GOPPhonemeClassifier.from_pretrained(
        model_repo_id, quantization_config=quantization_config, device_map="auto"
    )
    processor = Wav2Vec2Processor.from_pretrained(model_repo_id)
    model.eval()
    return model, processor


def process_audio_bytes(audio_bytes: bytes) -> torch.Tensor:
    """Converts raw bytes to a processed tensor ready for the model."""
    waveform, original_sr = torchaudio.load(BytesIO(audio_bytes))

    duration_seconds = waveform.shape[1] / original_sr
    if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
        raise ValueError(f"Audio too long. Max {MAX_AUDIO_DURATION_SECONDS}s allowed.")

    if waveform.shape[0] > MONO_CHANNEL:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if original_sr != SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sr, new_freq=SAMPLING_RATE
        )
        waveform = resampler(waveform)

    return waveform


def run_model_inference(waveform, phonemes, model, processor):
    """Runs the PyTorch inference for a single model."""
    device = model.device

    audio_input = waveform.squeeze(0)
    processed_audio = processor(
        audio_input, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True
    )
    input_values = processed_audio.input_values.to(device)
    attention_mask = processed_audio.attention_mask.to(device)

    tokenizer = processor.tokenizer
    unk_id = getattr(tokenizer, "unk_token_id", None)
    ids = tokenizer.convert_tokens_to_ids(phonemes)
    if isinstance(ids, int):
        ids = [ids]
    ids = [i if i is not None else unk_id for i in ids]

    canonical_token_ids = torch.tensor([ids], dtype=torch.long).to(device)
    token_lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)
    token_mask = torch.ones_like(canonical_token_ids).to(device)

    with torch.no_grad():
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            canonical_token_ids=canonical_token_ids,
            token_lengths=token_lengths,
            token_mask=token_mask,
        )

    logits = outputs.logits
    head_name = next(iter(logits))
    scores_tensor = logits[head_name]
    predicted_scores = torch.argmax(scores_tensor, dim=-1)

    return [int(s) + 1 for s in predicted_scores[0].cpu().tolist()]
