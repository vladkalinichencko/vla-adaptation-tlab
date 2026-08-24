# Адаптация SmolVLA к новым задачам LIBERO

Тестовое задание T-LAB, направление World, Action & Reward Models. Условие — в
[NOTES.md](NOTES.md), отчёт — в [report.md](report.md).

## Результаты

| метод | 5 демо | 10 демо | 25 демо |
|---|---:|---:|---:|
| подмешивание seen | 0.742 | 0.817 | 0.917 |
| LoRA r=32 | 0.575 | 0.717 | 0.858 |
| zero-shot | 0/60 | — | — |
| чужая инструкция | 0/60 | — | — |
| Bonus A, латентные действия | 0/60 | — | — |

Среднее по трём задачам `libero_goal` и двум training seeds, по 20 роллаутов на точку.
Seen-претрен: 5000 шагов на 450 эпизодах `libero_90`, loss 3.216 → 0.297.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

```bash
python -m vla.training      # обучение выбранного метода адаптации
python -m vla.evaluation    # роллауты и строка в runs/results.jsonl
python cost_curve.py        # кривая success против числа демонстраций
python viz.py               # -> report_page.html
python make_figures.py      # рисунок отчёта
```

## Раскладка кода

| путь | что там |
|---|---|
| `vla/data.py` | фиксированный split, первые демонстрации по порядку |
| `vla/training.py` | один training path на все методы адаптации |
| `vla/methods.py` | подмешивание seen, LoRA и наивный fine-tune |
| `vla/evaluation.py` | роллауты, success и запись в `runs/results.jsonl` |
| `vla/behavior.py`, `vla/diagnostics.py` | action chunks, контроль на язык, разбор фейлов |
| `vla/modeling_latent_smolvla.py` | латентные действия для Bonus A |
| `rollouts.py`, `cost_curve.py`, `viz.py` | сбор роллаутов, кривая и страница |

Интерактивная диагностика с реальными кадрами и роллаут-видео:
[report_page.html](report_page.html).
