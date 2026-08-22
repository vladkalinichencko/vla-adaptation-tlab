from vla.data import SEEN_CHECKPOINT, TARGET_INSTRUCTIONS
from vla.evaluation import evaluate, wrong_instruction
from vla.runtime import current_runtime


def run():
    runtime = current_runtime()
    rows = []
    for task_id in runtime.task_ids:
        instruction = TARGET_INSTRUCTIONS[(task_id + 1) % len(TARGET_INSTRUCTIONS)]
        with wrong_instruction(instruction):
            rows.append(
                evaluate(
                    f"wrong_instruction_t{task_id}",
                    SEEN_CHECKPOINT,
                    "wrong_instruction",
                    task_id,
                    0,
                    0,
                    runtime,
                    diagnostic_instruction=instruction,
                )
            )
    return rows


if __name__ == "__main__":
    run()
