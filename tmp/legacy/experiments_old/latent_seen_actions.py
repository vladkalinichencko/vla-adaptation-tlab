from pathlib import Path

from vla.data import SEEN_SOURCE, TARGET_SOURCE, dataset, first_target_episodes
from vla.evaluation import evaluate
from vla.runtime import adaptation_cells, current_runtime, training_steps
from vla.training import load_latent_policy, train_policy


TRANSITION = Path("outputs/latent_transition/checkpoints/last/pretrained_model")
SEEN_DECODER = Path("outputs/latent_seen_decoder/checkpoints/last/pretrained_model")


def train_seen_decoder(runtime):
    policy = load_latent_policy(TRANSITION, None, runtime, "action")
    steps = 1500 if runtime.is_screening else 30_000
    return train_policy("latent_seen_decoder", policy, dataset(SEEN_SOURCE), steps, 0, runtime)


def run():
    runtime = current_runtime()
    train_seen_decoder(runtime)
    rows = []
    for seed, task_id, demos in adaptation_cells(runtime):
        name = f"latent_seen_t{task_id}_n{demos}_s{seed}"
        policy = load_latent_policy(SEEN_DECODER, None, runtime, "action")
        data = dataset(TARGET_SOURCE, first_target_episodes(task_id, demos))
        checkpoint = train_policy(name, policy, data, training_steps(runtime, demos), seed, runtime)
        rows.append(evaluate(name, checkpoint, "latent_seen_actions", task_id, demos, seed, runtime))
    return rows


if __name__ == "__main__":
    run()
