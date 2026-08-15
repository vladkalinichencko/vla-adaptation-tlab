"""First N demos of a target task -> the `--dataset.episodes=[...]` list.

The demo budget must be the *first* episodes in dataset order, not a lucky pick.

В LeRobot v3.0 у метаданных эпизода нет колонки с задачей: `task_index` лежит в
самих данных. Карту «эпизод -> задача» строим один раз по паркетам (~1 минута на
lerobot/libero) и кешируем в datasets/.

    python demo_subset.py --list
    python demo_subset.py --task-index 10 --n 5

ВНИМАНИЕ: --task-index это индекс задачи в датасете, а НЕ --env.task_ids симулятора.
Порядок сьюта задаёт пакет LIBERO, сверьте инструкцию перед набором демо (NOTES).
"""

import argparse
import json
import pathlib

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

CACHE_DIR = pathlib.Path(__file__).parent / "datasets"


def episode_task_map(repo_id, revision=None, refresh=False):
    """-> {episode_index: task_index}, с кешем на диске."""
    cache = CACHE_DIR / f"episode_tasks_{repo_id.replace('/', '_')}.json"
    if cache.exists() and not refresh:
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}

    import pyarrow.dataset as ds

    root = f"hf://datasets/{repo_id}" + (f"@{revision}" if revision else "")
    print(f"строю карту эпизод -> задача по {root}/data (это разово, ~минуту)")
    table = ds.dataset(f"{root}/data", format="parquet").to_table(
        columns=["episode_index", "task_index"]
    )
    mapping = {}
    for episode, task in zip(table["episode_index"].to_pylist(), table["task_index"].to_pylist()):
        mapping.setdefault(int(episode), int(task))

    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(mapping))
    return mapping


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="lerobot/libero")
    p.add_argument("--revision", default=None)
    p.add_argument("--list", action="store_true", help="print task_index -> instruction")
    p.add_argument("--task", default=None, help="instruction string")
    p.add_argument("--task-index", type=int, default=None)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--refresh", action="store_true", help="пересобрать карту, игнорируя кеш")
    args = p.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id, revision=args.revision)

    if args.list:
        for task, row in meta.tasks.iterrows():
            print(f"{int(row.task_index):>3}  {task}")
        return

    if args.task is not None:
        task_index = meta.get_task_index(args.task)
        if task_index is None:
            raise SystemExit(f"нет такой задачи: {args.task!r}")
    elif args.task_index is not None:
        task_index = args.task_index
    else:
        raise SystemExit("нужен --task, --task-index или --list")

    instruction = meta.tasks.index[meta.tasks.task_index == task_index]
    if not len(instruction):
        raise SystemExit(f"нет задачи с индексом {task_index}")

    mapping = episode_task_map(args.repo_id, args.revision, args.refresh)
    episodes = [ep for ep in sorted(mapping) if mapping[ep] == task_index][: args.n]

    print(f"task_index {task_index}: {instruction[0]}")
    print(f"эпизодов у задачи: {sum(t == task_index for t in mapping.values())}")
    print(f"первые {len(episodes)}: {episodes}")
    print("--dataset.episodes=[" + ",".join(str(e) for e in episodes) + "]")


if __name__ == "__main__":
    main()
