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

Health check (API up):
```bash
curl -f http://localhost:8000/openapi.json
```

### Dockerfile notes

The Docker image is built on `python:3.12-slim`, installs `ffmpeg` for audio decoding, and uses `uv` to install Python deps from `pyproject.toml`/`uv.lock`. It then downloads the models during build (so runtime can be offline) and launches `uvicorn`.

### Hardware requirements (estimate)

These are approximate for CPU-only inference and depend on the actual model sizes.
- CPU: 2+ cores recommended (1 core works but concurrency will be limited).
- RAM: ~1 GB for 1 worker and ~2 GB for 2 workers on this build (from `docker stats`). Each `uvicorn` worker loads its own model copy, so RAM scales roughly linearly; verify on your hardware and with your workload.
- Storage: ~10.6 GB image size (includes models). Check actual size with `docker images` and `du -sh /models` inside the container.

#### Worker scaling guidance

Start with 1-2 workers unless you have a clear concurrent load profile. More workers increase RAM usage and can hurt latency if CPU is saturated.

To set workers:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

To scale workers in Docker, override the command:
```bash
docker run -p 8000:8000 sayest-api:latest \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

#### Storage sizing

You can measure the final image size and model cache:
```bash
docker images sayest-api:latest
```
```bash
docker run --rm sayest-api:latest du -sh /models
```
To estimate RAM per worker, run with 1 worker, then increase and compare:
```bash
docker stats <container>
```
