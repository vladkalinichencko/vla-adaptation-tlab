#!/usr/bin/env bash
# Попытка поставить LIBERO на macOS в обход hf-egl-probe.
#
# Логика: egl_probe нужен только для выбора GPU при headless-рендере через EGL,
# то есть под Linux/NVIDIA. На Darwin robosuite сам ставит MUJOCO_GL=cgl и EGL
# не трогает. Значит ставим всё дерево без egl_probe и смотрим, что сломается.
#
#   bash tmp/install_libero_macos.sh

set -uo pipefail
cd "$(dirname "$0")/.."
PIP=".venv/bin/pip"
PY=".venv/bin/python"

echo "### 1. зависимости robosuite, кроме egl_probe"
$PIP install -q numba scipy opencv-python pynput termcolor "mujoco>=3,<3.9" Pillow 2>&1 | tail -3

echo "### 2. robosuite и libero без своих зависимостей"
$PIP install -q --no-deps robosuite==1.4.0 2>&1 | tail -3
$PIP install -q --no-deps hf-libero 2>&1 | tail -3

echo "### 3. то, что libero тянет помимо egl_probe"
$PIP install -q --no-deps robomimic bddl 2>&1 | tail -3
$PIP install -q h5py easydict future thop 2>&1 | tail -3

echo
echo "### 4. что импортируется"
$PY - <<'PYEOF'
import os
os.environ.setdefault("MUJOCO_GL", "cgl")

for name in ("mujoco", "robosuite", "bddl", "robomimic", "libero"):
    try:
        mod = __import__(name)
        print(f"  OK      {name}")
    except Exception as exc:
        print(f"  ПАДАЕТ  {name}: {type(exc).__name__}: {str(exc)[:100]}")

print("\n### 5. список задач сьюта libero_goal")
try:
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    print(f"  задач в сьюте: {suite.n_tasks}")
    for i in range(min(3, suite.n_tasks)):
        print(f"  task_id={i}: {suite.get_task(i).language}")
except Exception as exc:
    print(f"  ПАДАЕТ: {type(exc).__name__}: {str(exc)[:200]}")
PYEOF
