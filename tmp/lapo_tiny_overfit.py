import gc
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
OUTPUT = Path("runs/lapo_pipeline_tiny_overfit")


def reconstruction_metrics(inverse_model, forward_model, examples):
    latents = [inverse_model(current, next_frame) for current, next_frame, _, _, _, _, _ in examples]
    shuffled = latents[1:] + latents[:1]
    totals = {"loss": 0.0, "zero_loss": 0.0, "shuffled_loss": 0.0, "cosine": 0.0, "latent_norm": 0.0}

    for (current, _, target, valid, _, _, _), latent, wrong_latent in zip(examples, latents, shuffled):
        predicted = forward_model(current, latent)
        zero = forward_model(current, torch.zeros_like(latent))
        wrong = forward_model(current, wrong_latent)
        mask = valid[:, :, None, None, None]
        count = (valid.sum() * target.shape[2] * target.shape[3] * target.shape[4]).clamp_min(1)
        totals["loss"] += ((((predicted - target) ** 2) * mask).sum() / count).item()
        totals["zero_loss"] += ((((zero - target) ** 2) * mask).sum() / count).item()
        totals["shuffled_loss"] += ((((wrong - target) ** 2) * mask).sum() / count).item()
        cosine = F.cosine_similarity(predicted.flatten(3), target.flatten(3), dim=-1)
        totals["cosine"] += ((cosine * valid[:, :, None]).sum() / (valid.sum() * target.shape[2])).item()
        totals["latent_norm"] += ((latent.norm(dim=-1) * valid).sum() / valid.sum()).item()

    return {name: value / len(examples) for name, value in totals.items()}


def policy_metrics(inverse_model, latent_policy, examples):
    losses = []
    cosines = []
    for current, next_frame, _, valid, language, _, _ in examples:
        target = inverse_model(current, next_frame)
        predicted = latent_policy(current[:, 0], language)
        loss = (((predicted - target) ** 2) * valid.unsqueeze(-1)).sum()
        losses.append((loss / (valid.sum() * target.shape[-1]).clamp_min(1)).item())
        cosine = F.cosine_similarity(predicted, target, dim=-1)
        cosines.append(((cosine * valid).sum() / valid.sum()).item())
    return {
        "loss": sum(losses) / len(losses),
        "cosine": sum(cosines) / len(cosines),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Diagnostic output already exists: {OUTPUT}")

    runtime = current_runtime()
    torch.manual_seed(0)
    config = load_latent_policy(BASE_POLICY, BASE_POLICY_REVISION, runtime, "representation")
    config.vision_encode_batch_size = 1
    source = dataset(SEEN_SOURCE, balanced_seen_episodes(1))
    setup = prepare_training("lapo_tiny_overfit", config, source, STEPS, 0, runtime)
    policy = setup.policy

    examples = []
    for index in INDICES:
        batch = setup.preprocessor(lerobot_collate_fn([setup.dataset[index]]))
        with torch.no_grad():
            visual = policy._visual_features(batch)
            current, next_frame = visual[:, :-1], visual[:, 1:]
            target = next_frame - current
            valid = policy._valid_transitions(batch, target.shape[1])
            language = policy._language_feature(batch)
            actions = pad_vector(batch["action"], policy.config.max_action_dim)
            action_valid = ~batch.get(
                "action_is_pad",
                torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device),
            )
            examples.append(
                tuple(
                    value.cpu()
                    for value in (current, next_frame, target, valid, language, actions, action_valid)
                )
            )
        print(f"encoded fixed index {index}", flush=True)
        if runtime.device == "mps":
            torch.mps.empty_cache()

    inverse_model = policy.inverse_model
    forward_model = policy.forward_model
    latent_policy = policy.latent_policy
    action_decoder = policy.action_decoder
    action_dim = policy.config.action_feature.shape[0]
    del setup, policy
    gc.collect()
    if runtime.device == "mps":
        torch.mps.empty_cache()

    dtype = next(inverse_model.parameters()).dtype
    examples = [
        tuple(
            value.to(runtime.device, dtype=dtype) if value.is_floating_point() else value.to(runtime.device)
            for value in example
        )
        for example in examples
    ]
    parameters = list(inverse_model.parameters()) + list(forward_model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LR)

    with torch.no_grad():
        initial = reconstruction_metrics(inverse_model, forward_model, examples)
    history = []
    for step in range(1, STEPS + 1):
        current, next_frame, target, valid, _, _, _ = examples[(step - 1) % len(examples)]
        latent = inverse_model(current, next_frame)
        predicted = forward_model(current, latent)
        mask = valid[:, :, None, None, None]
        count = (valid.sum() * target.shape[2] * target.shape[3] * target.shape[4]).clamp_min(1)
        loss = (((predicted - target) ** 2) * mask).sum() / count
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0:
            with torch.no_grad():
                metrics = reconstruction_metrics(inverse_model, forward_model, examples)
            history.append({"step": step, **metrics})
            print(json.dumps(history[-1]), flush=True)

    inverse_model.requires_grad_(False)
    forward_model.requires_grad_(False)
    latent_policy.requires_grad_(True)
    optimizer = torch.optim.AdamW(latent_policy.parameters(), lr=LR)
    policy_history = []
    for step in range(1, STEPS + 1):
        current, next_frame, _, valid, language, _, _ = examples[(step - 1) % len(examples)]
        with torch.no_grad():
            target_latent = inverse_model(current, next_frame)
        predicted_latent = latent_policy(current[:, 0], language)
        loss = (((predicted_latent - target_latent) ** 2) * valid.unsqueeze(-1)).sum()
        loss = loss / (valid.sum() * target_latent.shape[-1]).clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 10 == 0:
            with torch.no_grad():
                policy_history.append({
                    "step": step,
                    **policy_metrics(inverse_model, latent_policy, examples),
                })

    latent_policy.requires_grad_(False)
    action_decoder.requires_grad_(True)
    optimizer = torch.optim.AdamW(action_decoder.parameters(), lr=LR)
    for step in range(STEPS):
        current, next_frame, _, _, _, actions, action_valid = examples[step % len(examples)]
        with torch.no_grad():
            target_latent = inverse_model(current, next_frame)
        decoded = action_decoder(target_latent)
        loss = (((decoded - actions) ** 2) * action_valid.unsqueeze(-1)).sum()
        loss = loss / (action_valid.sum() * decoded.shape[-1]).clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    action_rows = []
    with torch.no_grad():
        for current, next_frame, _, valid, language, actions, action_valid in examples:
            target_latent = inverse_model(current, next_frame)
            predicted_latent = latent_policy(current[:, 0], language)
            cosine = F.cosine_similarity(predicted_latent, target_latent, dim=-1)
            action_rows.append({
                "latent_cosine": ((cosine * valid).sum() / valid.sum()).item(),
                "true_mae": ((action_decoder(target_latent)[..., :action_dim] - actions[..., :action_dim]).abs()
                             * action_valid.unsqueeze(-1)).sum().div(action_valid.sum() * action_dim).item(),
                "predicted_mae": ((action_decoder(predicted_latent)[..., :action_dim] - actions[..., :action_dim]).abs()
                                  * action_valid.unsqueeze(-1)).sum().div(action_valid.sum() * action_dim).item(),
                "zero_mae": ((action_decoder(torch.zeros_like(target_latent))[..., :action_dim] - actions[..., :action_dim]).abs()
                             * action_valid.unsqueeze(-1)).sum().div(action_valid.sum() * action_dim).item(),
            })

    def average(name):
        return sum(row[name] for row in action_rows) / len(action_rows)

    result = {
        "status": "completed",
        "method": "continuous LAPO-style inverse and forward model",
        "latent_dim": config.latent_dim,
        "fixed_episode": balanced_seen_episodes(1)[0],
        "fixed_indices": INDICES,
        "steps": STEPS,
        "lr": LR,
        "initial": initial,
        "final": history[-1],
        "latent_policy_final": policy_history[-1],
        "action_mae": {
            "true_latent": average("true_mae"),
            "predicted_latent": average("predicted_mae"),
            "zero_latent": average("zero_mae"),
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    (OUTPUT / "policy_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in policy_history)
    )
    (OUTPUT / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
