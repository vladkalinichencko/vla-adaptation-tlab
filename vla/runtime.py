from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Runtime:
    device: str
    batch_size: int
    workers: int
    mixed_precision: str
    eval_episodes: int
    task_ids: tuple[int, ...]
    budgets: tuple[int, ...]
    seeds: tuple[int, ...]

    @property
    def is_screening(self) -> bool:
        return self.device == "mps"


def current_runtime() -> Runtime:
    if torch.cuda.is_available():
        return Runtime("cuda", 32, 8, "bf16", 20, (0, 1, 2), (5, 10, 25), (0, 1))
    if torch.backends.mps.is_available():
        return Runtime("mps", 2, 4, "no", 5, (0,), (5,), (0,))
    raise RuntimeError("These experiments require MPS or CUDA.")


def training_steps(runtime: Runtime, demos: int) -> int:
    return 1500 if runtime.is_screening else 300 * demos


def adaptation_cells(runtime: Runtime):
    for seed in runtime.seeds:
        for task_id in runtime.task_ids:
            for demos in runtime.budgets:
                yield seed, task_id, demos
