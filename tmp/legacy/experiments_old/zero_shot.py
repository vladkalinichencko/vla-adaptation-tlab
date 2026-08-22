from vla.data import SEEN_CHECKPOINT
from vla.evaluation import evaluate
from vla.runtime import current_runtime


def run():
    runtime = current_runtime()
    return [
        evaluate(f"zero_shot_t{task_id}", SEEN_CHECKPOINT, "zero_shot", task_id, 0, 0, runtime)
        for task_id in runtime.task_ids
    ]


if __name__ == "__main__":
    run()
