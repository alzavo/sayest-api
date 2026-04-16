# sayest-api

API for pronunciation assessment. It accepts audio and a phoneme transcript, then returns per-phoneme scores for two heads from a single multihead model:

- `quality`
- `duration`

## Environment

Runtime variables:

- `MODEL_PATH` - required; local path or Hugging Face repo id for the multihead model
- `QUALITY_PROB_GAP_DELTA` - optional float; delta override for the `quality` head
- `DURATION_PROB_GAP_DELTA` - optional float; delta override for the `duration` head

Build-time variable:

- `MODEL_REPO_ID` - used by Docker build to download the model into the image

Create `.env` from the example:

```bash
cp .env.example .env
```

## Delta Behavior

Both heads support the same delta rule. If the model slightly prefers an incorrect class over the correct class, the API can still return score `1` when the probability gap is within the configured delta.

Example with `QUALITY_PROB_GAP_DELTA=0.02`:

```text
[[0.0, 0.03, -2.0], [0.0, 1.5, -2.0]]
```

Rule:

- compute softmax probabilities
- compare the probability of the correct class (`index 0`, score `1`) with the best incorrect class
- if the incorrect class wins by at most `delta`, return score `1`
- otherwise return the argmax score

For the first phoneme, the gap is small enough, so the score becomes `1`.
For the second phoneme, the gap is much larger, so the argmax score is used.

## Development

Install dependencies:

```bash
uv sync --dev
```

Run the API locally:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env
```

The app serves the FastAPI API.

For local development, `MODEL_PATH` can be set to the Hugging Face repo id
`alzavo/sayest-latest`. In that case, the app uses the downloaded model from
the local Hugging Face cache.

## Phoneme Vocabulary

The allowed phoneme set is stored in app/constants/phonemes.py.

## Tests

Run tests with:

```bash
uv run pytest
```

## Code Style

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

## Docker

Build the image:

```bash
docker build -t sayest-api:latest .
```

Build with a specific model repo:

```bash
docker build --build-arg MODEL_REPO_ID=alzavo/sayest-latest -t sayest-api:latest .
```

If you previously built an image that pulled CUDA PyTorch wheels into the slim base image, rebuild without cache:

```bash
docker build --no-cache --build-arg MODEL_REPO_ID=alzavo/sayest-latest -t sayest-api:latest .
```

### Docker Dependencies

The Docker image installs Python dependencies from `requirements-docker.txt`.

This file is generated specifically for the container runtime so the image uses
CPU-only PyTorch wheels:

- `torch==...+cpu`
- `torchaudio==...+cpu`

This avoids pulling large CUDA/NVIDIA packages into the image.

`requirements-docker.txt` is generated from `pyproject.toml` with:

```bash
uv pip compile pyproject.toml --no-emit-package sayest-api --torch-backend cpu -o requirements-docker.txt
```

If you change runtime dependencies in `pyproject.toml`, update both lock files
before rebuilding the image:

```bash
uv lock
uv pip compile pyproject.toml --no-emit-package sayest-api --torch-backend cpu -o requirements-docker.txt
docker build --no-cache -t sayest-api:latest .
```

Run the container:

```bash
docker run -p 8000:8000 sayest-api:latest
```

Run the container with a specific environment file:

```bash
docker run --env-file .env -p 8000:8000 sayest-api:latest
```

The runtime environment file is used for values such as `MODEL_PATH`,
`QUALITY_PROB_GAP_DELTA`, and `DURATION_PROB_GAP_DELTA`. If you want to change
which model gets downloaded into the image during the build, use the
`MODEL_REPO_ID` build argument shown above.

When running locally, `MODEL_PATH` can be a Hugging Face repo id such as
`alzavo/sayest-latest`, which resolves to the cached downloaded model. Inside
the Docker image, the model is downloaded into `/models`, so `MODEL_PATH`
should be `/models/<model-id>`, for example `/models/alzavo/sayest-latest`.

Health check:

```bash
curl -f http://localhost:8000/openapi.json
```

### Dockerfile Notes

The Docker image:

- starts from `python:3.12-slim`
- installs CPU-only `torch` and `torchaudio` wheels
- installs Python dependencies with `uv`
- downloads one multihead model during build by running `download_model.py`
- sets `HF_HUB_OFFLINE=1` so runtime can stay offline

## Hardware Notes

These are rough CPU-only estimates and depend on the actual model size and workload.

- CPU: 2+ cores recommended
- RAM: each `uvicorn` worker loads its own model copy, so memory scales roughly with worker count
- Storage: image size includes the downloaded model under `/models`
- Request handling is synchronous; for concurrent users, scale with additional `uvicorn` workers

To run more workers:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

To override the command in Docker:

```bash
docker run -p 8000:8000 sayest-api:latest \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```
