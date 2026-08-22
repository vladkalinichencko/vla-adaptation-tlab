"""Naive baseline for the cost curve: finetune the seen checkpoint on N target demos.

This is the reference point for Task 2 and must not be edited after the fact.
It is a thin wrapper over the official LeRobot CLI — see ext/lerobot/docs/source/libero.mdx.

    python baseline.py train --episodes 12 13 14 15 16 --tag base_t0_n5 --seed 0
    python baseline.py eval  --policy outputs/base_t0_n5/checkpoints/last/pretrained_model \\
                             --task-id 0 --method baseline --n-demos 5 --seed 0

Linux: MUJOCO_GL=egl, --device cuda. macOS: MUJOCO_GL=cgl, --device mps (медленнее,
но работает — см. setup_macos_libero.sh).
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

import mlflow

SUITE = "libero_goal"
RESULTS = pathlib.Path("runs") / "results.jsonl"

# smolvla ждёт camera1/2/3, LIBERO отдаёт image/image2 — нужно и train, и eval
RENAME_MAP = ('--rename_map={"observation.images.image": "observation.images.camera1", '
              '"observation.images.image2": "observation.images.camera2"}')


def aggregated(info):
    """Метрики из eval_info.json: lerobot кладёт их под 'overall' (или 'aggregated')."""
    for key in ("overall", "aggregated"):
        if key in info:
            block = info[key]
            return block.get("aggregated", block)
    raise KeyError(f"не нашёл агрегированные метрики, ключи: {list(info)}")


def record(row):
    """Одна строка в runs/results.jsonl и одна запись в MLflow. Все скрипты пишут сюда."""
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("vla-cost-curve")
    name = f"{row['method']}_n{row['n_demos']}_t{row['task']}_s{row['seed']}"
    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: row[k] for k in ("method", "seed", "task", "n_demos", "policy")})
        mlflow.log_metrics({"success": row["success"], "n_episodes": row["n_episodes"]},
                           step=row["n_demos"])
    print(f"success {row['success'] * 100:.1f}%  -> {RESULTS}")


def cli(name):
    """CLI из того же venv, что и текущий python — чтобы работало без активации."""
    candidate = pathlib.Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name


def run(cmd, dry):
    print(" \\\n  ".join(cmd))
    if dry:
        return 0
    return subprocess.call(cmd)


def cmd_train(a, extra=()):
    out = f"outputs/{a.tag}"
    cmd = [
        cli("lerobot-train"),
        f"--policy.path={a.ckpt}",
        f"--dataset.repo_id={a.dataset}",
        f"--output_dir={out}",
        f"--job_name={a.tag}",
        f"--steps={a.steps}",
        f"--batch_size={a.batch_size}",
        f"--seed={a.seed}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        RENAME_MAP,
    ]
    if a.device:
        cmd.append(f"--policy.device={a.device}")
    if a.episodes:
        cmd.append("--dataset.episodes=[" + ",".join(str(e) for e in a.episodes) + "]")
    if a.revision:
        cmd.append(f"--dataset.revision={a.revision}")
    cmd += list(extra)  # флаги приёмов адаптации уходят в lerobot как есть
    return run(cmd, a.dry_run)


def cmd_eval(a):
    out = f"eval_logs/{a.tag or f'{pathlib.Path(a.policy).parts[1]}_t{a.task_id}'}"
    cmd = [
        cli("lerobot-eval"),
        f"--policy.path={a.policy}",
        f"--output_dir={out}",
        "--env.type=libero",
        f"--env.task={a.suite}",
        f"--env.task_ids=[{a.task_id}]",
        f"--eval.n_episodes={a.n_episodes}",
        "--eval.batch_size=1",
        "--env.max_parallel_tasks=1",
        f"--seed={a.eval_seed}",
        "--env.init_states=true",
        RENAME_MAP,
    ]
    if a.device:
        cmd.append(f"--policy.device={a.device}")
    code = run(cmd, a.dry_run)
    if code or a.dry_run:
        return code

    info = json.loads((pathlib.Path(out) / "eval_info.json").read_text())
    record({
        "method": a.method,
        "seed": a.seed,
        "task": f"{a.suite}_{a.task_id}",
        "n_demos": a.n_demos,
        "success": aggregated(info)["pc_success"] / 100,
        "n_episodes": a.n_episodes,
        "policy": a.policy,
    })
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (по умолчанию решает lerobot)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--ckpt", default="lerobot/smolvla_base", help="seen checkpoint to adapt")
    t.add_argument("--dataset", default="lerobot/libero")
    t.add_argument("--revision", default=None, help="pin the dataset commit when reporting")
    t.add_argument("--episodes", type=int, nargs="*", help="first N demos, from demo_subset.py")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--tag", required=True)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--policy", required=True)
    e.add_argument("--suite", default=SUITE)
    e.add_argument("--task-id", type=int, required=True)
    e.add_argument("--n-episodes", type=int, default=20)
    e.add_argument("--eval-seed", type=int, default=1000)
    e.add_argument("--method", default="baseline", help="label written to results.jsonl")
    e.add_argument("--n-demos", type=int, required=True)
    e.add_argument("--seed", type=int, default=0, help="training seed of the evaluated policy")
    e.add_argument("--tag", default=None)
    e.set_defaults(fn=cmd_eval)

    a, extra = p.parse_known_args()
    raise SystemExit(a.fn(a, extra) if a.fn is cmd_train else a.fn(a))


if __name__ == "__main__":
    main()
