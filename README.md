# sayest-api

API for pronunciation assessment. Accepts audio + transcription (phonemes) and returns quality and duration scores.

## Environment

Required variables (use `.env` or export them):
- `QUALITY_MODEL_PATH` - path or repo id for the quality model
- `DURATION_MODEL_PATH` - path or repo id for the duration model

## Development

Create .env file:
```bash
cp .env.example .env
```

Install dependencies:
```bash
uv sync --dev
```

Run the API locally:
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env
```

## Tests

```bash
uv run pytest
```

## Code style (Ruff)

Check:
```bash
uv run ruff check .
```

Auto-fix:
```bash
uv run ruff check . --fix
```

Format check:
```bash
uv run ruff format --check .
```

## Production (Docker)

Build image:
```bash
docker build -t sayest-api:latest .
```

Run container:
```bash
docker run -p 8000:8000 sayest-api:latest
```

### Dockerfile notes

The Docker image is built on `python:3.12-slim`, installs `ffmpeg` for audio decoding, and uses `uv` to install Python deps from `pyproject.toml`/`uv.lock`. It then downloads the models during build (so runtime can be offline) and launches `uvicorn`.

Health check (API up):
```bash
curl -f http://localhost:8000/openapi.json
```
