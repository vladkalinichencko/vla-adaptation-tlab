import json
from pathlib import Path

import torch
import torch.nn.functional as F
from lerobot.policies.common.vla_utils import pad_vector
from lerobot.utils.collate import lerobot_collate_fn

from vla.data import BASE_POLICY, BASE_POLICY_REVISION, SEEN_SOURCE, balanced_seen_episodes, dataset
from vla.runtime import current_runtime
from vla.training import load_latent_policy, prepare_training


STEPS = 1000
LR = 1e-3
INDICES = (0, 10, 20)
OUTPUT = Path("runs/latent_tiny_overfit")


def measure_one(predicted, target):
    return {
        "loss": ((predicted - target) ** 2).mean().item(),
        "cosine": F.cosine_similarity(predicted.flatten(3), target.flatten(3), dim=-1).mean().item(),
    }


def measure(policy, examples):
    rows = [
        measure_one(policy.transition_head(current, language), target)
        for current, language, target, _ in examples
    ]
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Diagnostic output already exists: {OUTPUT}")
    runtime = current_runtime()
    config = load_latent_policy(BASE_POLICY, BASE_POLICY_REVISION, runtime, "transition")
    source = dataset(SEEN_SOURCE, balanced_seen_episodes(1))
    setup = prepare_training("latent_tiny_overfit", config, source, STEPS, 0, runtime)
    policy = setup.policy

    examples = []
    for index in INDICES:
        batch = setup.preprocessor(lerobot_collate_fn([setup.dataset[index]]))
        with torch.no_grad():
            visual = policy._visual_features(batch)
            current = visual[:, 0]
            language = policy._language_feature(batch)
            target = (visual[:, 1:] - visual[:, :-1]).to(current.dtype)
            actions = pad_vector(batch["action"], policy.config.max_action_dim).to(current.dtype)
            examples.append((current, language, target, actions))

    with torch.no_grad():
        initial = measure(policy, examples)

    optimizer = torch.optim.AdamW(policy.transition_head.parameters(), lr=LR)
    history = []
    for step in range(1, STEPS + 1):
        current, language, target, _ = examples[(step - 1) % len(examples)]
        predicted = policy.transition_head(current, language)
        loss = ((predicted - target) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 10 == 0:
            with torch.no_grad():
                history.append({"step": step, **measure(policy, examples)})

    decoder = policy.action_decoder
    decoder.requires_grad_(True)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=LR)
    for step in range(STEPS):
        _, _, target, actions = examples[step % len(examples)]
        decoder_input = target.mean(dim=3).flatten(2)
        decoded = decoder(decoder_input)
        loss = ((decoded - actions) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        action_rows = []
        for current, language, target, actions in examples:
            predicted = policy.transition_head(current, language)
            decoder_input = target.mean(dim=3).flatten(2)
            predicted_actions = decoder(predicted.mean(dim=3).flatten(2))
            true_actions = decoder(decoder_input)
            zero_actions = decoder(torch.zeros_like(decoder_input))
            reversed_actions = decoder(predicted.flip(1).mean(dim=3).flatten(2))
            action_rows.append({
                "true": (true_actions - actions).abs().mean().item(),
                "predicted": (predicted_actions - actions).abs().mean().item(),
                "zero": (zero_actions - actions).abs().mean().item(),
                "reversed": (reversed_actions - predicted_actions).abs().mean().item(),
            })
        def average(key):
            return sum(row[key] for row in action_rows) / len(action_rows)

        result = {
            "status": "completed",
            "fixed_episode": balanced_seen_episodes(1)[0],
            "fixed_indices": INDICES,
            "steps_per_phase": STEPS,
            "lr": LR,
            "initial_transition": initial,
            "final_transition": measure(policy, examples),
            "true_latent_action_mae": average("true"),
            "predicted_latent_action_mae": average("predicted"),
            "zero_latent_action_mae": average("zero"),
            "reversed_latent_action_change": average("reversed"),
        }

    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    (OUTPUT / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
