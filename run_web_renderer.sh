#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
export PYTHONPATH=.

uvicorn web.app:app --host 0.0.0.0 --port "${PORT:-10000}"