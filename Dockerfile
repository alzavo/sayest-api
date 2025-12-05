FROM python:3.12-slim

ARG QUALITY_MODEL_REPO_ID=alzavo/sayest-quality
ARG DURATION_MODEL_REPO_ID=alzavo/sayest-duration

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QUALITY_MODEL_REPO_ID=${QUALITY_MODEL_REPO_ID} \
    DURATION_MODEL_REPO_ID=${DURATION_MODEL_REPO_ID} \
    QUALITY_MODEL_PATH=/models/${QUALITY_MODEL_REPO_ID} \
    DURATION_MODEL_PATH=/models/${DURATION_MODEL_REPO_ID} \
    HF_HOME=/models

WORKDIR /code

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache .


COPY download_model.py .
RUN python download_model.py

ENV HF_HUB_OFFLINE=1

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
