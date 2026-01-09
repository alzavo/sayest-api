# sayest-api

## run container
docker run -p 8000:8000 sayest-api:latest

## build image
docker build -t sayest-api:latest .

## launch locally with .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
