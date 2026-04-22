#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/apps/rag}"
BRANCH="${BRANCH:-main}"
SKIP_GIT_SYNC="${SKIP_GIT_SYNC:-false}"

cd "$APP_DIR"

echo "[deploy] app dir: $APP_DIR"
echo "[deploy] branch: $BRANCH"

if [[ "$SKIP_GIT_SYNC" != "true" ]]; then
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
fi

# docker-compose v1 on Ubuntu 22.04 can fail to recreate an existing container
# with a stale ContainerConfig. Remove the old API container first to keep
# deployments idempotent.
sudo docker rm -f rag-api >/dev/null 2>&1 || true

sudo docker-compose -f docker-compose.api.yml up -d --build api

for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
    echo "[deploy] success"
    exit 0
  fi
  sleep 2
done

echo "[deploy] health check timed out" >&2
exit 1
