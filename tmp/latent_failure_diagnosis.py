import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from lerobot.utils.collate import lerobot_collate_fn

from vla.data import BASE_POLICY, BASE_POLICY_REVISION, SEEN_SOURCE, balanced_seen_episodes, dataset
from vla.modeling_latent_smolvla import TransitionHead
from vla.runtime import current_runtime
from vla.training import load_latent_policy, prepare_training


STEPS = 1000
LR = 1e-3
INDICES = (0, 10, 20)
OUTPUT = Path("runs/latent_failure_diagnosis")


def metrics(head, examples, pooled):
    rows = []
    for current, language, target in examples:
        predicted = head(current, language)
        if pooled:
            predicted = predicted.mean(dim=3)
            target = target.mean(dim=3)
        rows.append({
            "loss": ((predicted - target) ** 2).mean().item(),
            "cosine": F.cosine_similarity(predicted.flatten(2), target.flatten(2), dim=-1).mean().item(),
            "predicted_norm": predicted.norm(dim=-1).mean().item(),
            "target_norm": target.norm(dim=-1).mean().item(),
        })
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def fit(name, examples, pooled, hidden, device, dtype):
    torch.manual_seed(0)
    head = TransitionHead(hidden, 50, 2).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
    history = [{"step": 0, **metrics(head, examples, pooled)}]
    for step in range(1, STEPS + 1):
        current, language, target = examples[(step - 1) % len(examples)]
        predicted = head(current, language)
        if pooled:
            predicted = predicted.mean(dim=3)
            target = target.mean(dim=3)
        loss = ((predicted - target) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            with torch.no_grad():
                history.append({"step": step, **metrics(head, examples, pooled)})
    return {
        "name": name,
        "examples": len(examples),
        "pooled_loss": pooled,
        "initial": history[0],
        "final": history[-1],
        "history": history,
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Diagnostic output already exists: {OUTPUT}")
    runtime = current_runtime()
    config = load_latent_policy(BASE_POLICY, BASE_POLICY_REVISION, runtime, "transition")
    config.vision_encode_batch_size = 1
    source = dataset(SEEN_SOURCE, balanced_seen_episodes(1))
    setup = prepare_training("latent_failure_diagnosis", config, source, STEPS, 0, runtime)
    policy = setup.policy

    examples = []
    for index in INDICES:
        batch = setup.preprocessor(lerobot_collate_fn([setup.dataset[index]]))
        with torch.no_grad():
            visual = policy._visual_features(batch)
            examples.append(tuple(value.cpu() for value in (
                visual[:, 0],
                policy._language_feature(batch),
                visual[:, 1:] - visual[:, :-1],
            )))
        del batch, visual
        if runtime.device == "mps":
            torch.mps.empty_cache()

    hidden = policy.transition_head.step_embedding.shape[-1]
    dtype = next(policy.transition_head.parameters()).dtype
    del setup, policy
    gc.collect()
    if runtime.device == "mps":
        torch.mps.empty_cache()
    examples = [tuple(value.to(runtime.device, dtype=dtype) for value in row) for row in examples]

    results = [
        fit("one_window_tokenwise", examples[:1], False, hidden, runtime.device, dtype),
        fit("three_windows_spatial_mean", examples, True, hidden, runtime.device, dtype),
    ]
    OUTPUT.mkdir(parents=True)
    for result in results:
        path = OUTPUT / f"{result['name']}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in result.pop("history")))
    (OUTPUT / "run.json").write_text(json.dumps({
        "status": "completed",
        "indices": INDICES,
        "steps": STEPS,
        "lr": LR,
        "tests": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
