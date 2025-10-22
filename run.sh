#!/usr/bin/env bash
set -e
docker compose up -d
# Ingest
python -m app.ingest
# Jalankan API
uvicorn app.api:app --reload --port 8000