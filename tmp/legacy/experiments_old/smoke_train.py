from vla.data import BASE_POLICY, BASE_POLICY_REVISION, SEEN_SOURCE, balanced_seen_episodes, dataset
from vla.runtime import current_runtime
from vla.training import load_policy, train_policy


def run():
    runtime = current_runtime()
    policy = load_policy(BASE_POLICY, BASE_POLICY_REVISION, runtime)
    data = dataset(SEEN_SOURCE, balanced_seen_episodes(2))
    return train_policy("smoke_train", policy, data, 10, 0, runtime)


if __name__ == "__main__":
    run()
