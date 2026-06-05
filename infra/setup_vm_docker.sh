#!/usr/bin/env bash
# Docker-based vLLM-on-TPU setup. Run on the TPU VM after provision_tpu.sh.
# Replaces the brittle pip-install path (which hits torch/torch_xla/vllm version hell).
#
# What this installs:
#   - docker (if missing)
#   - postgres (for Person 2's SQL executor)
#   - vllm-tpu Docker image, pre-pulled so first 'serve' is fast
#
# What it does NOT do:
#   - python venv (not needed — vllm runs inside container)
#   - jax / torch / torch_xla on host (not needed — runs inside container)
set -euo pipefail

echo "==> apt installs (postgres, tmux, htop, docker prerequisites)"
sudo apt-get update -y
sudo apt-get install -y postgresql postgresql-contrib tmux htop curl ca-certificates

echo "==> Installing Docker if missing"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sudo sh /tmp/get-docker.sh
  sudo usermod -aG docker "$USER"
  # New group membership takes effect in new shells; use sudo for the rest of this session.
fi
sudo docker --version

echo "==> Pulling vllm-tpu image (~3 GB, one-time)"
sudo docker pull vllm/vllm-tpu:nightly

echo "==> Configuring Postgres (sqlagent role + spider_eval db)"
sudo systemctl enable --now postgresql
sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='sqlagent'" \
  | grep -q 1 || sudo -u postgres createuser -s sqlagent
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='spider_eval'" \
  | grep -q 1 || sudo -u postgres createdb -O sqlagent spider_eval

echo "==> Done. Next: bash serve_vllm.sh"
