import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from lerobot.configs.default import EvalConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.scripts.lerobot_eval import eval_main

from vla.behavior import action_snapshot
from vla.data import RENAME, TARGET_SOURCE, TARGET_SUITE, first_target_episodes
from vla.diagnostics import append_result
from vla.observer import log_evaluation
from vla.runtime import Runtime


def evaluate(
    name: str,
    checkpoint: str | Path,
    method: str,
    task_id: int,
    demos: int,
    seed: int,
    runtime: Runtime,
    diagnostic_instruction: str | None = None,
) -> dict:
    output = Path("eval_logs") / name
    info_path = output / "eval_info.json"
    if output.exists() and not info_path.is_file():
        raise FileExistsError(f"Incomplete eval output already exists: {output}")

    if not info_path.is_file():
        policy = PreTrainedConfig.from_pretrained(checkpoint)
        policy.pretrained_path = Path(checkpoint)
        policy.device = runtime.device
        config = EvalPipelineConfig(
            env=LiberoEnv(task=TARGET_SUITE, task_ids=[task_id], init_states=True, max_parallel_tasks=1),
            eval=EvalConfig(n_episodes=runtime.eval_episodes, batch_size=1, use_async_envs=False),
            policy=policy,
            output_dir=output,
            job_name=name,
            seed=1000,
            rename_map=RENAME,
        )
        eval_main(config)
    info = json.loads(info_path.read_text())
    success = info["overall"]["pc_success"] / 100
    row = {
        "method": method,
        "seed": seed,
        "task": f"{TARGET_SUITE}_{task_id}",
        "n_demos": demos,
        "success": success,
        "n_episodes": runtime.eval_episodes,
        "policy": str(checkpoint),
        "eval": str(output / "eval_info.json"),
    }
    row["action_diagnostics"] = str(
        action_snapshot(
            name,
            checkpoint,
            TARGET_SOURCE,
            first_target_episodes(task_id, max(demos, 1)),
            runtime,
            diagnostic_instruction,
        )
    )
    append_result(row)
    log_evaluation(name, row, runtime.device)
    return row


@contextmanager
def wrong_instruction(instruction: str) -> Iterator[None]:
    from lerobot.envs import libero

    original = libero.LiberoEnv.__init__

    def patched(environment, *args, **kwargs):
        original(environment, *args, **kwargs)
        environment.task_description = instruction

    libero.LiberoEnv.__init__ = patched
    try:
        yield
    finally:
        libero.LiberoEnv.__init__ = original
