# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ARG MODEL_REPO_ID=alzavo/sayest-latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_REPO_ID=${MODEL_REPO_ID} \
    MODEL_PATH=/models/${MODEL_REPO_ID} \
    HF_HOME=/models

WORKDIR /code

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    .

COPY download_model.py .
RUN python download_model.py

ENV HF_HUB_OFFLINE=1

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
