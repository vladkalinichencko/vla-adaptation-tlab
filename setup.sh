#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

[ -d ext/lerobot ] || git clone --depth 1 https://github.com/huggingface/lerobot.git ext/lerobot
[ -d .venv ] || python3 -m venv .venv

.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "ok: source .venv/bin/activate"
