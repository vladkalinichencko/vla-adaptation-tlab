from vla.data import SEEN_CHECKPOINT, TARGET_SOURCE, dataset, first_target_episodes
from vla.evaluation import evaluate
from vla.runtime import adaptation_cells, current_runtime, training_steps
from vla.training import load_policy, train_policy


def run():
    runtime = current_runtime()
    rows = []
    for seed, task_id, demos in adaptation_cells(runtime):
        name = f"longer_t{task_id}_n{demos}_s{seed}"
        policy = load_policy(SEEN_CHECKPOINT, None, runtime)
        data = dataset(TARGET_SOURCE, first_target_episodes(task_id, demos))
        steps = 2 * training_steps(runtime, demos)
        checkpoint = train_policy(name, policy, data, steps, seed, runtime)
        rows.append(evaluate(name, checkpoint, "longer_finetune", task_id, demos, seed, runtime))
    return rows


if __name__ == "__main__":
    run()
