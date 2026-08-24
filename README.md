# World, Action & Reward Models

Адаптация SmolVLA к новым задачам LIBERO по пяти, десяти и двадцати пяти демонстрациям.

Отчёт — [report.md](report.md). Диагностика с кадрами и видео роллаутов — [report_page.html](report_page.html).

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m vla.training     # дообучение выбранным методом
python -m vla.evaluation   # роллауты и строка в runs/results.jsonl
python cost_curve.py       # кривая успеха против числа демонстраций
python viz.py              # пересборка report_page.html
```

## Где что лежит

| из отчёта | в коде |
|---|---|
| подмешивание знакомых задач, LoRA, наивное дообучение | `vla/methods.py` |
| латентные действия, бонусная часть | `vla/modeling_latent_smolvla.py`, `vla/configuration_latent_smolvla.py` |
| фиксированный split и порядок демонстраций | `vla/data.py` |
| общий путь обучения | `vla/training.py` |
| роллауты, успех, запись результатов | `vla/evaluation.py`, `rollouts.py` |
| действия, контроль на язык, разбор неудач | `vla/behavior.py`, `vla/diagnostics.py` |
| кривая стоимости | `cost_curve.py` |
| сборка страницы | `viz.py` |

Итоговые артефакты и веса — в `runs/final/`.
