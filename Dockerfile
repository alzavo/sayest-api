# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ARG MODEL_REPO_ID=alzavo/sayest-latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_REPO_ID=${MODEL_REPO_ID} \
    MODEL_PATH=/models/${MODEL_REPO_ID} \
    HF_HOME=/models

WORKDIR /code

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
COPY requirements-docker.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --exact --requirement requirements-docker.txt --torch-backend cpu

COPY download_model.py .
RUN python download_model.py

ENV HF_HUB_OFFLINE=1

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
