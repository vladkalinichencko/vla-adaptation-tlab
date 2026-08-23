import gc
import json
from pathlib import Path

import torch

from vla.behavior import augmentation_snapshot, transition_snapshot
from vla.data import (
    BASE_POLICY,
    BASE_POLICY_REVISION,
    SEEN_SOURCE,
    TARGET_INSTRUCTIONS,
    TARGET_SOURCE,
    balanced_seen_episodes,
    build_mix,
    dataset,
    first_target_episodes,
)
from vla.evaluation import evaluate, wrong_instruction
from vla.methods import apply_lora, use_action_chunk, use_full_finetune, use_image_augmentations
from vla.runtime import current_runtime
from vla.training import load_latent_policy, load_policy, prepare_training, train


STEPS = 100
LONG_STEPS = 200
LATENT_STEPS = 50
WARMUP_STEPS = 10
BASE_LR = 1e-4
FINAL_LR = 2.5e-6


def fit(name, policy_config, data_config, runtime, steps=STEPS, lr=BASE_LR):
    checkpoint = completed_checkpoint(name)
    if checkpoint:
        return checkpoint
    setup = prepare_training(name, policy_config, data_config, steps, 0, runtime)
    checkpoint = train(
        setup,
        runtime,
        lr=lr,
        warmup_steps=WARMUP_STEPS,
        final_lr=FINAL_LR,
    )
    del setup
    gc.collect()
    if runtime.device == "mps":
        torch.mps.empty_cache()
    return checkpoint


def completed_checkpoint(name):
    path = Path("runs") / name / "run.json"
    if not path.is_file():
        return None
    row = json.loads(path.read_text())
    checkpoint = Path(row.get("checkpoint", ""))
    if row.get("status") == "completed" and (checkpoint / "config.json").is_file():
        return checkpoint
    return None


def evaluate_once(name, checkpoint, method, runtime, instruction=None):
    info = Path("eval_logs") / name / "eval_info.json"
    actions = Path("runs/diagnostics") / f"{name}_actions.json"
    if info.is_file() and actions.is_file():
        return
    evaluate(name, checkpoint, method, 0, 5 if "zero_shot" not in name and "wrong_instruction" not in name else 0, 0, runtime, instruction)


def adapt(method, seen_checkpoint: Path, runtime):
    name = f"preliminary_{method}_t0_n5_s0"
    checkpoint = completed_checkpoint(name)
    if checkpoint:
        evaluate_once(name, checkpoint, method, runtime)
        return
    policy_config = load_policy(seen_checkpoint, None, runtime)
    data_config = dataset(TARGET_SOURCE, first_target_episodes(0, 5))
    steps = STEPS
    lr = BASE_LR

    if method == "longer_finetune":
        steps = LONG_STEPS
    elif method == "full_finetune":
        policy_config = use_full_finetune(policy_config)
    elif method == "mix_seen":
        mix = build_mix(0, 5)
        data_config = dataset(mix.source)
        steps = round(STEPS * mix.total_frames / mix.target_frames)
    elif method == "lora_r32":
        lr = 1e-3
    elif method == "chunk_10":
        policy_config = use_action_chunk(policy_config, 10)
    elif method == "image_augmentations":
        data_config = use_image_augmentations(data_config)
    elif method != "naive_finetune":
        raise ValueError(f"Unknown preliminary method: {method}")

    setup = prepare_training(name, policy_config, data_config, steps, 0, runtime)
    if method == "lora_r32":
        setup.policy = apply_lora(setup.policy, rank=32)
    checkpoint = train(
        setup,
        runtime,
        lr=lr,
        warmup_steps=WARMUP_STEPS,
        final_lr=FINAL_LR,
    )
    evaluate_once(name, checkpoint, method, runtime)
    del setup
    gc.collect()
    torch.mps.empty_cache()


def run_latent(runtime):
    representation = fit(
        "preliminary_lapo_representation",
        load_latent_policy(BASE_POLICY, BASE_POLICY_REVISION, runtime, "representation"),
        dataset(SEEN_SOURCE),
        runtime,
        LATENT_STEPS,
    )
    transition_snapshot(
        "preliminary_lapo_representation",
        representation,
        SEEN_SOURCE,
        balanced_seen_episodes(1),
        runtime,
    )

    latent_policy = fit(
        "preliminary_lapo_policy",
        load_latent_policy(representation, None, runtime, "policy"),
        dataset(SEEN_SOURCE),
        runtime,
        LATENT_STEPS,
    )
    seen_decoder = fit(
        "preliminary_lapo_seen_decoder",
        load_latent_policy(latent_policy, None, runtime, "action"),
        dataset(SEEN_SOURCE),
        runtime,
        LATENT_STEPS,
    )
    target = dataset(TARGET_SOURCE, first_target_episodes(0, 5))

    checkpoint = fit(
        "preliminary_lapo_seen_actions_t0_n5_s0",
        load_latent_policy(seen_decoder, None, runtime, "action"),
        target,
        runtime,
    )
    evaluate_once("preliminary_lapo_seen_actions_t0_n5_s0", checkpoint, "lapo_seen_actions", runtime)

    checkpoint = fit(
        "preliminary_lapo_video_only_t0_n5_s0",
        load_latent_policy(latent_policy, None, runtime, "action"),
        target,
        runtime,
    )
    evaluate_once("preliminary_lapo_video_only_t0_n5_s0", checkpoint, "lapo_video_only", runtime)


def main():
    runtime = current_runtime()
    if not runtime.is_screening:
        raise RuntimeError("This file is only for the short MPS screening pass.")

    augmentation_snapshot(TARGET_SOURCE, first_target_episodes(0, 1))

    seen_checkpoint = fit(
        "preliminary_seen_pretrain",
        load_policy(BASE_POLICY, BASE_POLICY_REVISION, runtime),
        dataset(SEEN_SOURCE),
        runtime,
    )
    evaluate_once("preliminary_zero_shot_t0", seen_checkpoint, "zero_shot", runtime)

    instruction = TARGET_INSTRUCTIONS[1]
    with wrong_instruction(instruction):
        evaluate_once(
            "preliminary_wrong_instruction_t0",
            seen_checkpoint,
            "wrong_instruction",
            runtime,
            instruction,
        )

    for method in (
        "naive_finetune",
        "longer_finetune",
        "full_finetune",
        "mix_seen",
        "lora_r32",
        "chunk_10",
        "image_augmentations",
    ):
        adapt(method, seen_checkpoint, runtime)

    run_latent(runtime)


if __name__ == "__main__":
    main()
