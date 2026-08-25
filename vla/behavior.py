import json
from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION
from torchvision.utils import save_image

from vla.data import RENAME, Source, metadata
from vla.diagnostics import plain
from vla.runtime import Runtime

def augmentation_snapshot(source: Source, episodes: list[int]) -> Path:
    data = LeRobotDataset(
        source.repo_id,
        root=source.root,
        revision=source.revision,
        episodes=episodes,
        video_backend="pyav",
    )
    item = data[0]
    image = item["observation.images.image"]
    config = ImageTransformsConfig(enable=True)
    pipeline = ImageTransforms(config)
    names = {id(transform): name for name, transform in pipeline.transforms.items()}
    output = Path("runs/diagnostics/augmentations")
    output.mkdir(parents=True, exist_ok=True)
    save_image(image, output / "original.png")

    samples = [{"name": "original", "image": "original.png", "transforms": []}]
    for seed in range(4):
        torch.manual_seed(seed)
        augmented = pipeline(image)
        filename = f"seed_{seed}.png"
        save_image(augmented, output / filename)
        samples.append({
            "name": f"seed {seed}",
            "image": filename,
            "transforms": [names[id(transform)] for transform in pipeline.tf.selected_transforms],
        })

    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps({
        "episode": int(item["episode_index"]),
        "frame": int(item["frame_index"]),
        "max_num_transforms": config.max_num_transforms,
        "available": {name: {"type": value.type, "range": value.kwargs}
                      for name, value in config.tfs.items()},
        "samples": samples,
    }, indent=2) + "\n")
    return metadata_path

def action_snapshot(
    name: str,
    checkpoint: str | Path,
    source: Source,
    episodes: list[int],
    runtime: Runtime,
    instruction: str | None = None,
) -> Path:
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
            if hasattr(policy, "latent_policy"):
                visual = policy._visual_features(batch)
                latent = policy.latent_policy(visual[:, 0], policy._language_feature(batch))
                action_dim = policy.config.action_feature.shape[0]
                decoded = policy.action_decoder(latent)[..., :action_dim]
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
            if hasattr(policy, "latent_policy"):
                zero = policy.action_decoder(torch.zeros_like(latent))[..., :action_dim]
                reversed_steps = policy.action_decoder(latent.flip(1))[..., :action_dim]
                true_latent = policy.inverse_model(visual[:, :-1], visual[:, 1:]).to(
                    policy.action_decoder.weight.dtype
                )
                true_latent_actions = policy.action_decoder(true_latent)[..., :action_dim]
                row.update(
                    latent_norm=latent.norm(dim=-1)[0].tolist(),
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
        current, next_frame = visual[:, :-1], visual[:, 1:]
        latent = policy.inverse_model(current, next_frame)
        predicted = policy.forward_model(current, latent)
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
                "latent_norm_by_step": latent.norm(dim=-1)[0].cpu().tolist(),
                "raw_tensors": str(raw),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return path
