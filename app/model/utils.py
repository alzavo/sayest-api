import torch
import torchaudio
from io import BytesIO
from transformers import Wav2Vec2Processor
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
    model = GOPPhonemeClassifier.from_pretrained(model_repo_id, device_map="auto")
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


def parse_delta_value(value):
    if value is None or value == "":
        return None
    try:
        delta = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid delta value %r; ignoring.", value)
        return None
    if delta <= 0:
        return None
    return delta


def _predict_scores(scores_tensor, delta=None, correct_index=0):
    num_classes = scores_tensor.size(-1)
    use_delta = delta is not None and delta > 0 and 0 <= correct_index < num_classes

    if use_delta:
        probs = torch.softmax(scores_tensor, dim=-1)
        correct_probs = probs[..., correct_index]
        incorrect_probs = probs.clone()
        incorrect_probs[..., correct_index] = -float("inf")
        max_incorrect_probs, _ = incorrect_probs.max(dim=-1)
        argmax_scores = probs.argmax(dim=-1)
        within_delta = (max_incorrect_probs > correct_probs) & (
            (max_incorrect_probs - correct_probs) <= delta
        )
        predicted_scores = torch.where(
            within_delta,
            torch.tensor(correct_index, device=scores_tensor.device),
            argmax_scores,
        )
    else:
        if delta is not None and (correct_index < 0 or correct_index >= num_classes):
            logger.warning(
                "Delta provided but correct_index=%s is out of range for %s classes.",
                correct_index,
                num_classes,
            )
        predicted_scores = torch.argmax(scores_tensor, dim=-1)

    return [int(score) + 1 for score in predicted_scores[0].cpu().tolist()]


def run_model_inference(waveform, phonemes, model, processor, deltas=None, correct_index=0):
    """Runs the PyTorch inference for the multihead model."""
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

    if not isinstance(outputs.logits, dict):
        raise TypeError("Expected multihead logits dict from model output.")

    head_deltas = deltas or {}
    return {
        head_name: _predict_scores(
            scores_tensor,
            delta=head_deltas.get(head_name),
            correct_index=correct_index,
        )
        for head_name, scores_tensor in outputs.logits.items()
    }
