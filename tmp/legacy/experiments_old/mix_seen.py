from vla.data import SEEN_CHECKPOINT, build_mix, dataset
from vla.evaluation import evaluate
from vla.runtime import adaptation_cells, current_runtime, training_steps
from vla.training import load_policy, train_policy


def run():
    runtime = current_runtime()
    rows = []
    for seed, task_id, demos in adaptation_cells(runtime):
        name = f"mix_t{task_id}_n{demos}_s{seed}"
        mix = build_mix(task_id, demos)
        baseline_steps = training_steps(runtime, demos)
        steps = round(baseline_steps * mix.total_frames / mix.target_frames)
        policy = load_policy(SEEN_CHECKPOINT, None, runtime)
        checkpoint = train_policy(name, policy, dataset(mix.source), steps, seed, runtime)
        rows.append(evaluate(name, checkpoint, "mix_seen", task_id, demos, seed, runtime))
    return rows


if __name__ == "__main__":
    run()
