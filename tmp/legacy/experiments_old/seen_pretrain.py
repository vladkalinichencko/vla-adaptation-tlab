from vla.data import BASE_POLICY, BASE_POLICY_REVISION, SEEN_SOURCE, dataset
from vla.runtime import current_runtime
from vla.training import load_policy, train_policy


def run():
    runtime = current_runtime()
    policy = load_policy(BASE_POLICY, BASE_POLICY_REVISION, runtime)
    steps = 1500 if runtime.is_screening else 30_000
    return train_policy("seen_pretrain", policy, dataset(SEEN_SOURCE), steps, 0, runtime)


if __name__ == "__main__":
    run()
