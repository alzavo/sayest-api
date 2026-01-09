from fastapi import FastAPI
import gradio as gr
from .api.routes import router
from .core.lifespan import lifespan
from .core.logging import setup_logging
from .ui import create_gradio_app

setup_logging()

app = FastAPI(
    title="SayEst API",
    description="API for the pronunciation assessment model. Accepts audio with transcription and returns scores.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)

demo = create_gradio_app(app).queue(default_concurrency_limit=2)
app = gr.mount_gradio_app(app, demo, path="/")
