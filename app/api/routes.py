from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from app.api.schemas import PredictionResponse, ErrorResponse
from app.constants.phonemes import ALL_PHONEMES
from app.model.utils import process_audio_bytes, run_model_inference
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_phonemes(
    request: Request,
    phonemes: str = Form(..., description="Space-separated phonemes"),
    word: str = Form(None),
    audio: UploadFile = File(...)
):
    # 1. Validation
    phoneme_list = phonemes.strip().split()
    if not phoneme_list:
        raise HTTPException(status_code=400, detail="Phonemes string is empty.")
    
    for p in phoneme_list:
        if p not in ALL_PHONEMES:
             raise HTTPException(status_code=400, detail=f"Invalid phoneme: {p}")

    if not hasattr(request.app.state, "models") or not request.app.state.models:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    # 2. Audio Processing
    try:
        content = await audio.read()
        waveform = process_audio_bytes(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise HTTPException(status_code=422, detail="Invalid audio file.")

    # 3. Inference
    try:
        q_model, q_proc = request.app.state.models["quality"]
        d_model, d_proc = request.app.state.models["duration"]
        
        # Run sequentially
        q_scores = run_model_inference(waveform, phoneme_list, q_model, q_proc)
        d_scores = run_model_inference(waveform, phoneme_list, d_model, d_proc)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal model error.")

    # 4. Response Construction
    details = []
    for i, p in enumerate(phoneme_list):
        q_val = q_scores[i] if i < len(q_scores) else -1
        d_val = d_scores[i] if i < len(d_scores) else -1
        
        details.append({
            "phoneme": p,
            "quality_score": q_val,
            "duration_score": d_val,
        })

    return {
        "success": True,
        "word": word,
        "original_transcript": phonemes,
        "details": details
    }
