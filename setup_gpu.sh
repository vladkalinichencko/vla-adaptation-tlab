#!/usr/bin/env bash
# Linux + GPU: полная установка LeRobot с LIBERO и SmolVLA.
set -euo pipefail
cd "$(dirname "$0")"

[ -d ext/lerobot ] || git clone --depth 1 https://github.com/huggingface/lerobot.git ext/lerobot
[ -d .venv ] || python3 -m venv .venv

.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -e "ext/lerobot[smolvla,libero,training]"

echo 'ok: source .venv/bin/activate && export MUJOCO_GL=egl'
