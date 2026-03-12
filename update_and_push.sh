#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 src/data/update.py

git add data/raw/fixtures/upcoming.json data/raw/odds/upcoming_odds.json data/processed/
git commit -m "data: update jornada $(date +%Y-%m-%d)"
git push
