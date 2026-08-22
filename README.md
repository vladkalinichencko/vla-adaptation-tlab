# SmolVLA на LIBERO: адаптация малым бюджетом демо

Тестовое задание T-LAB, World/Action/Reward Models. Условие — [NOTES.md](NOTES.md),
конвенции репозитория — [AGENTS.md](AGENTS.md).

## Setup

macOS (симулятор работает, но медленно — 20 шагов/с):

```bash
./setup.sh
./setup_macos_libero.sh      # LIBERO в обход hf-egl-probe
export MUJOCO_GL=cgl
```

Linux + GPU (полные прогоны):

```bash
./setup_gpu.sh
export MUJOCO_GL=egl
```

## Запуск

```bash
python run_preliminary.py
python viz.py
```

`run_preliminary.py` последовательно запускает короткий MPS-отсев. Методы собраны в
`vla/methods.py`, явный optimizer loop находится в `vla/training.py`. Параметры
написаны в Python, CLI-флагов обучения нет.

Каждый шаг пишется в `runs/<run>/metrics.jsonl`, конфигурация и статус в
`runs/<run>/run.json`. Оценки лежат в `runs/results.jsonl`, action chunks и latent
transitions в `runs/diagnostics/`. На CUDA те же значения дополнительно уходят в
ClearML. W&B и MLflow не используются.
