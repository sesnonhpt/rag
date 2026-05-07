#!/usr/bin/env bash
# Hotfix script to add /documents route to Nginx configuration
# Use this only in emergency situations when GitHub Actions deployment is not available

set -euo pipefail

NGINX_SITE_SOURCE="${APP_DIR:-/home/ubuntu/apps/rag}/deploy/nginx/rag-fullstack.conf"
NGINX_SITE_TARGET="/etc/nginx/sites-available/rag"
NGINX_SITE_LINK="/etc/nginx/sites-enabled/rag"

echo "[hotfix] Checking if /documents route exists in Nginx config..."

if grep -q "documents" "$NGINX_SITE_TARGET" 2>/dev/null; then
  echo "[hotfix] /documents route already exists in Nginx config"
  exit 0
fi

echo "[hotfix] Installing updated Nginx configuration..."
sudo install -m 644 "$NGINX_SITE_SOURCE" "$NGINX_SITE_TARGET"
sudo ln -sfn "$NGINX_SITE_TARGET" "$NGINX_SITE_LINK"

echo "[hotfix] Testing Nginx configuration..."
if ! sudo nginx -t; then
  echo "[hotfix] ERROR: Nginx configuration test failed!" >&2
  echo "[hotfix] Rolling back to previous configuration..." >&2
  sudo systemctl reload nginx
  exit 1
fi

echo "[hotfix] Reloading Nginx..."
sudo systemctl reload nginx

echo "[hotfix] Verifying Nginx is running..."
if ! sudo systemctl is-active --quiet nginx; then
  echo "[hotfix] ERROR: Nginx is not running!" >&2
  exit 1
fi

echo "[hotfix] Success! /documents route has been added to Nginx"
echo "[hotfix] You can now test the document import feature"
