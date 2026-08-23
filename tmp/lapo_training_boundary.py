import gc
from dataclasses import replace

import torch

from vla.data import BASE_POLICY, BASE_POLICY_REVISION, SEEN_SOURCE, balanced_seen_episodes, dataset
from vla.runtime import current_runtime
from vla.training import load_latent_policy, prepare_training, train


def fit_phase(name, checkpoint, revision, phase, runtime):
    config = load_latent_policy(checkpoint, revision, runtime, phase)
    if runtime.device == "mps":
        config.vision_encode_batch_size = 1
    setup = prepare_training(
        name,
        config,
        dataset(SEEN_SOURCE, balanced_seen_episodes(1)),
        1,
        0,
        runtime,
    )
    checkpoint = train(setup, runtime, lr=1e-3, warmup_steps=0, final_lr=1e-3)
    del setup
    gc.collect()
    if runtime.device == "mps":
        torch.mps.empty_cache()
    return checkpoint


def main():
    runtime = current_runtime()
    if runtime.device == "mps":
        runtime = replace(runtime, batch_size=1, workers=0)
    representation = fit_phase(
        "lapo_boundary_representation",
        BASE_POLICY,
        BASE_POLICY_REVISION,
        "representation",
        runtime,
    )
    policy = fit_phase("lapo_boundary_policy", representation, None, "policy", runtime)
    fit_phase("lapo_boundary_action", policy, None, "action", runtime)


if __name__ == "__main__":
    main()
