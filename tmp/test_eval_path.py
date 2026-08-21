"""Проверка ветки `baseline.py eval` без LIBERO: подсовываем готовый eval_info.json.

LIBERO не ставится на macOS, но разбор результата и запись в results.jsonl / MLflow
проверить можно и здесь. Запускать из корня проекта:  python tmp/test_eval_path.py
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import baseline  # noqa: E402

TAG = "faketest"
TASK_ID = 0


def main():
    out = pathlib.Path("eval_logs") / f"{TAG}_t{TASK_ID}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_info.json").write_text(json.dumps({
        "per_episode": [],
        "aggregated": {"avg_sum_reward": 0.4, "pc_success": 40.0, "eval_s": 1.0},
    }))

    baseline.run = lambda cmd, dry: print("(запуск подменён)", cmd[0]) or 0

    args = argparse.Namespace(
        policy="outputs/faketest/checkpoints/last/pretrained_model",
        suite="libero_goal", task_id=TASK_ID, n_episodes=20, eval_seed=1000,
        method="baseline", n_demos=5, seed=0, tag=TAG, dry_run=False,
    )
    print("cmd_eval ->", baseline.cmd_eval(args))
    print("последняя строка results.jsonl:")
    print(baseline.RESULTS.read_text().splitlines()[-1])


if __name__ == "__main__":
    main()
