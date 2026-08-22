"""Does the policy read the instruction at all?

Task 1 asks for a language control: run the same episodes with another task's
instruction. If success does not move, the policy is not conditioning on language and
the whole "zero-shot from one instruction" point of the assignment is empty for it.

Runs the official lerobot-eval, with LiberoEnv.task_description replaced after the env
is built — so everything else (init states, seeds, episode count) is identical to the
normal eval and only the sentence the policy reads changes.

    python language_control.py --policy outputs/fixed_n25/checkpoints/last/pretrained_model \\
        --task-id 0 --instruction "open the top drawer of the cabinet"
"""

import argparse
import json
import pathlib
import runpy
import sys

import baseline


def patch(instruction):
    from lerobot.envs import libero

    original = libero.LiberoEnv.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.task_description = instruction

    libero.LiberoEnv.__init__ = __init__


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--instruction", required=True, help="инструкция от другой задачи")
    p.add_argument("--suite", default=baseline.SUITE)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--eval-seed", type=int, default=1000)
    p.add_argument("--device", default="mps")
    p.add_argument("--n-demos", type=int, default=0, help="сколько демо видел чекпойнт")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    patch(args.instruction)
    sys.argv = ["lerobot-eval",
                f"--policy.path={args.policy}",
                f"--policy.device={args.device}",
                f"--output_dir=eval_logs/{args.tag}",
                "--env.type=libero",
                f"--env.task={args.suite}",
                f"--env.task_ids=[{args.task_id}]",
                f"--eval.n_episodes={args.n_episodes}",
                "--eval.batch_size=1",
                "--env.max_parallel_tasks=1",
                f"--seed={args.eval_seed}",
                "--env.init_states=true",
                baseline.RENAME_MAP]
    try:
        runpy.run_module("lerobot.scripts.lerobot_eval", run_name="__main__")
    except SystemExit as exit_:
        if exit_.code:
            raise

    info = json.loads((pathlib.Path("eval_logs") / args.tag / "eval_info.json").read_text())
    baseline.record({
        "method": "wrong_instruction",
        "seed": 0,
        "task": f"{args.suite}_{args.task_id}",
        "n_demos": args.n_demos,
        "success": baseline.aggregated(info)["pc_success"] / 100,
        "n_episodes": args.n_episodes,
        "policy": args.policy,
        "instruction": args.instruction,
    })


if __name__ == "__main__":
    main()
