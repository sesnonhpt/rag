#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/apps/rag}"
BRANCH="${BRANCH:-main}"

cd "$APP_DIR"

echo "[deploy] app dir: $APP_DIR"
echo "[deploy] branch: $BRANCH"

git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

docker-compose -f docker-compose.api.yml up -d --build api
curl -fsS http://127.0.0.1:8000/health >/dev/null

echo "[deploy] success"
