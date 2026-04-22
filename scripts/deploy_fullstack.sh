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

# Stop and remove old containers
echo "[deploy] stopping old containers..."
sudo docker-compose -f docker-compose.fullstack.yml down --remove-orphans >/dev/null 2>&1 || true
sudo docker ps -aq --filter "name=rag-" | xargs -r sudo docker rm -f >/dev/null 2>&1 || true

# Build and start new containers
echo "[deploy] building and starting containers..."
sudo docker-compose -f docker-compose.fullstack.yml up -d --build

# Wait for backend health check
echo "[deploy] waiting for backend health check..."
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "[deploy] backend is healthy"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "[deploy] backend health check timed out" >&2
    exit 1
  fi
  sleep 2
done

# Wait for frontend
echo "[deploy] waiting for frontend..."
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8080 >/dev/null 2>&1; then
    echo "[deploy] frontend is ready"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "[deploy] frontend check timed out" >&2
    exit 1
  fi
  sleep 2
done

echo "[deploy] full stack deployment successful"
exit 0
