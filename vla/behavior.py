import json
from pathlib import Path

import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION

from vla.data import RENAME, Source, metadata
from vla.diagnostics import plain
from vla.runtime import Runtime


def action_snapshot(
    name: str,
    checkpoint: str | Path,
    source: Source,
    episodes: list[int],
    runtime: Runtime,
    instruction: str | None = None,
) -> Path:
    from lerobot.configs.policies import PreTrainedConfig

    checkpoint = Path(checkpoint)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.device = runtime.device
    meta = metadata(source)
    data = LeRobotDataset(
        source.repo_id,
        root=source.root,
        revision=source.revision,
        episodes=episodes,
        delta_timestamps=resolve_delta_timestamps(config, meta, RENAME),
        video_backend="pyav",
    )
    policy = make_policy(config, ds_meta=data.meta, rename_map=RENAME).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": runtime.device},
            "rename_observations_processor": {"rename_map": RENAME},
        },
    )

    snapshots = []
    for index in (0, 10, 20):
        item = data[index]
        if instruction is not None:
            item["task"] = instruction
        policy.reset()
        torch.manual_seed(0)
        with torch.no_grad():
            batch = preprocessor(lerobot_collate_fn([item]))
            if hasattr(policy, "transition_head"):
                visual = policy._visual_features(batch)
                latent = policy._predict_transitions(visual, policy._language_feature(batch))
                action_dim = policy.config.action_feature.shape[0]
                decoded = policy.action_decoder(latent.mean(dim=3).flatten(2))[..., :action_dim]
                predicted = postprocessor(decoded)[0]
            else:
                predicted = postprocessor(policy.predict_action_chunk(batch))[0]
            row = {
                "episode": int(item["episode_index"]),
                "frame": int(item["frame_index"]),
                "instruction": item["task"],
                "target": item[ACTION].tolist(),
                "predicted": predicted.tolist(),
                "absolute_error_by_step": (predicted - item[ACTION]).abs().mean(dim=-1).tolist(),
            }
            if hasattr(policy, "transition_head"):
                zero = policy.action_decoder(torch.zeros_like(latent).mean(dim=3).flatten(2))[..., :action_dim]
                reversed_steps = policy.action_decoder(latent.flip(1).mean(dim=3).flatten(2))[..., :action_dim]
                true_latent = (visual[:, 1:] - visual[:, :-1]).to(policy.action_decoder.weight.dtype)
                true_latent_actions = policy.action_decoder(true_latent.mean(dim=3).flatten(2))[..., :action_dim]
                row.update(
                    latent_norm=latent.norm(dim=-1).mean(dim=(2, 3))[0].tolist(),
                    true_latent_actions=postprocessor(true_latent_actions)[0].tolist(),
                    zero_latent_actions=postprocessor(zero)[0].tolist(),
                    reversed_latent_actions=postprocessor(reversed_steps)[0].tolist(),
                )
        snapshots.append(row)

    path = Path("runs/diagnostics") / f"{name}_actions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(snapshots), indent=2, ensure_ascii=False) + "\n")
    return path


def transition_snapshot(
    name: str,
    checkpoint: str | Path,
    source: Source,
    episodes: list[int],
    runtime: Runtime,
) -> Path:
    from lerobot.configs.policies import PreTrainedConfig

    checkpoint = Path(checkpoint)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.device = runtime.device
    meta = metadata(source)
    data = LeRobotDataset(
        source.repo_id,
        root=source.root,
        revision=source.revision,
        episodes=episodes,
        delta_timestamps=resolve_delta_timestamps(config, meta, RENAME),
        video_backend="pyav",
    )
    policy = make_policy(config, ds_meta=data.meta, rename_map=RENAME).eval()
    preprocessor, _ = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": runtime.device},
            "rename_observations_processor": {"rename_map": RENAME},
        },
    )
    item = data[20]
    with torch.no_grad():
        batch = preprocessor(lerobot_collate_fn([item]))
        visual = policy._visual_features(batch)
        predicted = policy._predict_transitions(visual, policy._language_feature(batch))
        target = (visual[:, 1:] - visual[:, :-1]).to(predicted.dtype)
        cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean(dim=-1)

    raw = Path("runs/diagnostics") / f"{name}_transitions.pt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"target": target.cpu().half(), "predicted": predicted.cpu().half()}, raw)
    path = raw.with_suffix(".json")
    path.write_text(
        json.dumps(
            {
                "episode": int(item["episode_index"]),
                "frame": int(item["frame_index"]),
                "instruction": item["task"],
                "target_norm_by_step_and_view": target.norm(dim=-1).mean(dim=-1)[0].cpu().tolist(),
                "predicted_norm_by_step_and_view": predicted.norm(dim=-1).mean(dim=-1)[0].cpu().tolist(),
                "cosine_by_step_and_view": cosine[0].cpu().tolist(),
                "raw_tensors": str(raw),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return path
