"""First N demos of a target task -> the `--dataset.episodes=[...]` list.

The demo budget must be the *first* episodes in dataset order, not a lucky pick.

    python demo_subset.py --list
    python demo_subset.py --task-index 3 --n 5
"""

import argparse

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="lerobot/libero")
    p.add_argument("--revision", default=None)
    p.add_argument("--list", action="store_true", help="print task_index -> instruction")
    p.add_argument("--task", default=None, help="instruction string")
    p.add_argument("--task-index", type=int, default=None)
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id, revision=args.revision)

    if args.list:
        for task, row in meta.tasks.iterrows():
            print(f"{int(row.task_index):>3}  {task}")
        return

    task = args.task
    if task is None:
        if args.task_index is None:
            raise SystemExit("pass --task or --task-index (or --list)")
        matches = meta.tasks.index[meta.tasks.task_index == args.task_index]
        if not len(matches):
            raise SystemExit(f"no task with index {args.task_index}")
        task = matches[0]

    episodes = meta.filter_episodes(lambda ep: task in ep["tasks"])[: args.n]
    print(f"task: {task}")
    print(f"episodes ({len(episodes)}): {episodes}")
    print("--dataset.episodes=[" + ",".join(str(e) for e in episodes) + "]")


if __name__ == "__main__":
    main()
