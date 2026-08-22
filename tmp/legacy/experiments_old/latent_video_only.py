from pathlib import Path

from vla.data import TARGET_SOURCE, dataset, first_target_episodes
from vla.evaluation import evaluate
from vla.runtime import adaptation_cells, current_runtime, training_steps
from vla.training import load_latent_policy, train_policy


TRANSITION = Path("outputs/latent_transition/checkpoints/last/pretrained_model")


def run():
    runtime = current_runtime()
    rows = []
    for seed, task_id, demos in adaptation_cells(runtime):
        name = f"latent_video_t{task_id}_n{demos}_s{seed}"
        policy = load_latent_policy(TRANSITION, None, runtime, "action")
        data = dataset(TARGET_SOURCE, first_target_episodes(task_id, demos))
        checkpoint = train_policy(name, policy, data, training_steps(runtime, demos), seed, runtime)
        rows.append(evaluate(name, checkpoint, "latent_video_only", task_id, demos, seed, runtime))
    return rows


if __name__ == "__main__":
    run()
