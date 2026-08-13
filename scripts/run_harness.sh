#!/bin/bash
# Wrapper for the launchd/cron scheduled harness run. Does not require the
# FastAPI server to be running — talks directly to the SQLite DB and pulls
# from ESPN itself. Safe to run manually any time you want an on-demand
# refresh: `./scripts/run_harness.sh`
set -e
cd "$(dirname "$0")/.."
python3 backend/harness.py
