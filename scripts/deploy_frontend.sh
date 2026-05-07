#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/apps/rag}"
BRANCH="${BRANCH:-main}"
SKIP_GIT_SYNC="${SKIP_GIT_SYNC:-false}"

cd "$APP_DIR"

echo "[deploy-frontend] app dir: $APP_DIR"
echo "[deploy-frontend] branch: $BRANCH"

if [[ "$SKIP_GIT_SYNC" != "true" ]]; then
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
fi

build_and_start_frontend() {
  sudo docker-compose -f docker-compose.fullstack.yml up -d --build frontend
}

run_compose_with_output() {
  local log_file="$1"
  set +e
  build_and_start_frontend 2>&1 | tee "$log_file"
  local statuses=(${PIPESTATUS[@]})
  local compose_status="${statuses[0]}"
  set -e
  return "$compose_status"
}

# Stop and remove old frontend container
echo "[deploy-frontend] stopping old frontend container..."
sudo docker-compose -f docker-compose.fullstack.yml stop frontend >/dev/null 2>&1 || true
sudo docker ps -aq --filter "name=rag-frontend" | xargs -r sudo docker rm -f >/dev/null 2>&1 || true

# Build and start new frontend container
echo "[deploy-frontend] building and starting frontend..."
compose_log="$(mktemp)"
trap 'rm -f "$compose_log"' EXIT

if run_compose_with_output "$compose_log"; then
  compose_status=0
else
  compose_status=$?
fi

if [[ $compose_status -ne 0 ]]; then
  if grep -q 'container name "/rag-frontend" is already in use' "$compose_log"; then
    echo "[deploy-frontend] detected stale rag-frontend container name conflict, cleaning and retrying once..."
    sudo docker ps -aq --filter "name=^/rag-frontend$" | xargs -r sudo docker rm -f >/dev/null 2>&1 || true
    if run_compose_with_output "$compose_log"; then
      compose_status=0
    else
      compose_status=$?
    fi
  fi
fi

if [[ $compose_status -ne 0 ]]; then
  exit $compose_status
fi

# Wait for frontend
echo "[deploy-frontend] waiting for frontend..."
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8080 >/dev/null 2>&1; then
    echo "[deploy-frontend] frontend is ready"
    echo "[deploy-frontend] deployment successful"
    exit 0
  fi
  if [ $i -eq 30 ]; then
    echo "[deploy-frontend] frontend check timed out" >&2
    exit 1
  fi
  sleep 2
done
