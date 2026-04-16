from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from app.api.schemas import PredictionResponse, ErrorResponse
from app.constants.phonemes import ALL_PHONEMES
from app.constants.environmental_variables import (
    QUALITY_PROB_GAP_DELTA,
    DURATION_PROB_GAP_DELTA,
)
from app.model.utils import process_audio_bytes, run_model_inference, parse_delta_value
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def predict_phonemes(
    request: Request,
    phonemes: str = Form(..., description="Space-separated phonemes"),
    word: str = Form(None),
    audio: UploadFile = File(...),
):
    phoneme_list = phonemes.strip().split()
    if not phoneme_list:
        raise HTTPException(status_code=400, detail="Phonemes string is empty.")

    for p in phoneme_list:
        if p not in ALL_PHONEMES:
            raise HTTPException(status_code=400, detail=f"Invalid phoneme: {p}")

    if not hasattr(request.app.state, "model_artifacts"):
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        content = audio.file.read()
        waveform = process_audio_bytes(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise HTTPException(status_code=422, detail="Invalid audio file.")

    try:
        model, processor = request.app.state.model_artifacts
        scores_by_head = run_model_inference(
            waveform,
            phoneme_list,
            model,
            processor,
            {
                "quality": parse_delta_value(QUALITY_PROB_GAP_DELTA),
                "duration": parse_delta_value(DURATION_PROB_GAP_DELTA),
            },
            0,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal model error.")

    q_scores = scores_by_head.get("quality")
    d_scores = scores_by_head.get("duration")
    if q_scores is None or d_scores is None:
        raise HTTPException(
            status_code=500, detail="Model output is missing required heads."
        )

    details = []
    for i, p in enumerate(phoneme_list):
        q_val = q_scores[i]
        d_val = d_scores[i]

        details.append(
            {
                "phoneme": p,
                "quality_score": q_val,
                "duration_score": d_val,
            }
        )

    return {
        "success": True,
        "word": word,
        "original_transcript": phonemes,
        "details": details,
    }
