from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
from app.constants.environmental_variables import (
    MODEL_PATH,
)
from app.model.utils import load_model_and_processor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Lifespan: Loading model...")
    try:
        model, processor = load_model_and_processor(MODEL_PATH)

        app.state.model_artifacts = (model, processor)
        logger.info("Lifespan: Model loaded successfully.")
        yield
    except Exception as e:
        logger.error(f"Lifespan: Failed to load model: {e}")
        raise

    finally:
        if hasattr(app.state, "model_artifacts"):
            del app.state.model_artifacts
            logger.info("Lifespan: Model unloaded.")
