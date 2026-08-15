# SmolVLA на LIBERO: адаптация малым бюджетом демо

Тестовое задание T-LAB, World/Action/Reward Models. Условие — [NOTES.md](NOTES.md),
конвенции репозитория — [AGENTS.md](AGENTS.md).

## Setup

Локально (macOS) — только инспекция датасета и графики:

```bash
./setup.sh
source .venv/bin/activate
```

Обучение и оценка — на Linux-машине с GPU (LIBERO не ставится на macOS):

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
