from fastapi import FastAPI
from .api.routes import router
from .core.lifespan import lifespan
from .core.logging import setup_logging

setup_logging()

app = FastAPI(
    title="SayEst API",
    description="API for the pronunciation assessment model. Accepts audio with transcription and returns scores.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
