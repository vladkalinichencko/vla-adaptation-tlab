# AGENTS.md

Тестовое задание T-LAB, направление **World, Action & Reward Models**.
Постановка целиком — в [NOTES.md](NOTES.md). Прочитай её перед любой работой.

## Раскладка

| путь | что там | в git |
|---|---|---|
| `NOTES.md` | задание, сетап, грабли, предсказания, лог, результаты | да |
| `baseline.py` | наивный файнтюн на 5/10/25 демо + eval, обёртка над LeRobot CLI | да |
| `demo_subset.py` | первые N демо целевой задачи -> `--dataset.episodes=[...]` | да |
| `cost_curve.py` | `runs/results.jsonl` -> кривая success vs число демо | да |
| `report.md` | отчёт, **максимум 4 страницы** | да |
| `outputs/` | чекпойнты `lerobot-train` | нет |
| `eval_logs/` | `eval_info.json` от `lerobot-eval` | нет |
| `datasets/` | локальный кэш LeRobot-датасетов | нет |
| `runs/` | `results.jsonl`, графики, кадры фейлов | нет |
| `tmp/` | всё, что нагенерил агент | нет |
| `ext/lerobot` | клон LeRobot, только читать (там же документация по LIBERO) | нет |

## Правила

- **Бейзлайн замораживается.** Числа из `baseline.py` на 5/10/25 — точка отсчёта для
  Задачи 2, менять их задним числом нельзя.
- **Демо берём первые по порядку** (`demo_subset.py`), не выбираем удачные.
- Демо других целевых задач использовать **нельзя**; демо seen-задач — можно;
  собирать новые данные в среде — нельзя, среда только для оценки.
- Каждая точка кривой: 3 целевые задачи $\times \ge 20$ эпизодов $\times \ge 2$ сида.
- **Предсказания пишутся в NOTES ДО запусков.** Это оценивается отдельно, и задним
  числом не восстанавливается.
- Все запуски пишут строку в `runs/results.jsonl` (это делает `baseline.py eval`).
- Всё черновое — в `tmp/`.
- На macOS запускается только `--dry-run` и графики; обучение/оценка — на GPU-боксе.

## Команды

```bash
source .venv/bin/activate
python demo_subset.py --list
python demo_subset.py --task-index <i> --n 5
python baseline.py train --episodes <...> --tag base_t0_n5 --seed 0 --dry-run
python baseline.py eval --policy outputs/base_t0_n5/checkpoints/last/pretrained_model \
                        --task-id 0 --method baseline --n-demos 5 --seed 0
python cost_curve.py runs/results.jsonl --plot
```

## Что сдаём

- [ ] Запускаемый репозиторий с README: setup + точные команды на каждую задачу
- [ ] Отчёт **≤ 4 страниц**: предсказания, cost curve, разбор фейлов, бонус, Задача 5
- [ ] Три характерных фейла с кадрами + дизайн различающих экспериментов
- [ ] Один бонус на выбор (A: action-free, B: оценка без реварда, C: длинный горизонт)
