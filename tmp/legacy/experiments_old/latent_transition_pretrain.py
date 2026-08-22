from vla.behavior import transition_snapshot
from vla.data import BASE_POLICY, BASE_POLICY_REVISION, SEEN_SOURCE, balanced_seen_episodes, dataset
from vla.runtime import current_runtime
from vla.training import load_latent_policy, train_policy


def run():
    runtime = current_runtime()
    policy = load_latent_policy(BASE_POLICY, BASE_POLICY_REVISION, runtime, "transition")
    steps = 50 if runtime.is_screening else 30_000
    checkpoint = train_policy("latent_transition", policy, dataset(SEEN_SOURCE), steps, 0, runtime)
    transition_snapshot("latent_transition", checkpoint, SEEN_SOURCE, balanced_seen_episodes(1), runtime)
    return checkpoint


if __name__ == "__main__":
    run()
