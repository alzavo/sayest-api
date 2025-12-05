from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
from app.constants.environmental_variables import (
    QUALITY_MODEL_PATH,
    DURATION_MODEL_PATH,
)
from app.model.utils import load_model_and_processor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Lifespan: Loading models...")
    try:
        q_model, q_proc = load_model_and_processor(QUALITY_MODEL_PATH)
        d_model, d_proc = load_model_and_processor(DURATION_MODEL_PATH)

        app.state.models = {"quality": (q_model, q_proc), "duration": (d_model, d_proc)}
        logger.info("Lifespan: Models loaded successfully.")
        yield
    except Exception as e:
        logger.error(f"Lifespan: Failed to load models: {e}")


    finally:
        if hasattr(app.state, "models"):
            app.state.models.clear()
            logger.info("Lifespan: Models unloaded.")
