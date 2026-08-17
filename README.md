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

## Cost curve

```bash
python demo_subset.py --task-index <i> --n 5          # первые 5 демо целевой задачи
python baseline.py train --episodes <...> --tag base_t0_n5 --seed 0
python baseline.py eval --policy outputs/base_t0_n5/checkpoints/last/pretrained_model \
                        --task-id 0 --method baseline --n-demos 5 --seed 0
python cost_curve.py runs/results.jsonl --plot
```

Все точки копятся в `runs/results.jsonl`; график — `runs/cost_curve.png`.

Метрики — в локальный MLflow (`sqlite:///mlflow.db`):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
