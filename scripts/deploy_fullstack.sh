#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/apps/rag}"
BRANCH="${BRANCH:-main}"
SKIP_GIT_SYNC="${SKIP_GIT_SYNC:-false}"
NGINX_SITE_SOURCE="${APP_DIR}/deploy/nginx/rag-fullstack.conf"
NGINX_SITE_TARGET="/etc/nginx/sites-available/rag"
NGINX_SITE_LINK="/etc/nginx/sites-enabled/rag"

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

# Ensure host nginx routes frontend and backend on port 80.
if [[ -f "$NGINX_SITE_SOURCE" ]]; then
  echo "[deploy] installing nginx site config..."
  sudo install -m 644 "$NGINX_SITE_SOURCE" "$NGINX_SITE_TARGET"
  sudo ln -sfn "$NGINX_SITE_TARGET" "$NGINX_SITE_LINK"
  sudo nginx -t
  sudo systemctl reload nginx
fi

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

# Wait for public gateway on port 80.
echo "[deploy] waiting for public /lesson route..."
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1/lesson >/dev/null 2>&1; then
    echo "[deploy] public gateway is ready"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "[deploy] public gateway check timed out" >&2
    exit 1
  fi
  sleep 2
done

echo "[deploy] full stack deployment successful"
exit 0
