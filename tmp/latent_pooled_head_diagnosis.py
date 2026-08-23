import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from lerobot.utils.collate import lerobot_collate_fn
from torch import nn

from vla.data import BASE_POLICY, BASE_POLICY_REVISION, SEEN_SOURCE, balanced_seen_episodes, dataset
from vla.runtime import current_runtime
from vla.training import load_latent_policy, prepare_training


STEPS = 1000
LR = 1e-3
INDICES = (0, 10, 20)
OUTPUT = Path("runs/latent_pooled_head_diagnosis")


class PooledTransitionHead(nn.Module):
    def __init__(self, hidden, steps, views):
        super().__init__()
        self.step_embedding = nn.Parameter(torch.zeros(steps, hidden))
        self.view_embedding = nn.Parameter(torch.zeros(views, hidden))
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, current, language):
        current = current.mean(dim=2)
        hidden = (
            current[:, None]
            + language[:, None, None]
            + self.step_embedding[None, :, None]
            + self.view_embedding[None, None]
        )
        return self.predictor(hidden)


def metrics(head, examples):
    rows = []
    for current, language, target in examples:
        predicted = head(current, language)
        rows.append({
            "loss": ((predicted - target) ** 2).mean().item(),
            "cosine": F.cosine_similarity(predicted, target, dim=-1).mean().item(),
            "predicted_norm": predicted.norm(dim=-1).mean().item(),
            "target_norm": target.norm(dim=-1).mean().item(),
        })
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Diagnostic output already exists: {OUTPUT}")
    runtime = current_runtime()
    config = load_latent_policy(BASE_POLICY, BASE_POLICY_REVISION, runtime, "transition")
    config.vision_encode_batch_size = 1
    source = dataset(SEEN_SOURCE, balanced_seen_episodes(1))
    setup = prepare_training("latent_pooled_head_diagnosis", config, source, STEPS, 0, runtime)
    policy = setup.policy

    examples = []
    for index in INDICES:
        batch = setup.preprocessor(lerobot_collate_fn([setup.dataset[index]]))
        with torch.no_grad():
            visual = policy._visual_features(batch)
            examples.append(tuple(value.cpu() for value in (
                visual[:, 0],
                policy._language_feature(batch),
                (visual[:, 1:] - visual[:, :-1]).mean(dim=3),
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

    torch.manual_seed(0)
    head = PooledTransitionHead(hidden, 50, 2).to(device=runtime.device, dtype=dtype)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
    history = [{"step": 0, **metrics(head, examples)}]
    for step in range(1, STEPS + 1):
        current, language, target = examples[(step - 1) % len(examples)]
        predicted = head(current, language)
        loss = ((predicted - target) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            with torch.no_grad():
                history.append({"step": step, **metrics(head, examples)})

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    (OUTPUT / "run.json").write_text(json.dumps({
        "status": "completed",
        "indices": INDICES,
        "steps": STEPS,
        "lr": LR,
        "initial": history[0],
        "final": history[-1],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
