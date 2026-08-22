import dataclasses
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.collate import lerobot_collate_fn
from torch.utils.data import DataLoader, RandomSampler

from vla.configuration_latent_smolvla import LatentSmolVLAConfig
from vla.data import RENAME
from vla.diagnostics import plain
from vla.observer import TrainingObserver
from vla.runtime import Runtime


@dataclass
class TrainingSetup:
    config: TrainPipelineConfig
    dataset: Any
    policy: PreTrainedPolicy
    preprocessor: Any
    postprocessor: Any


def load_policy(checkpoint: str | Path, revision: str | None, runtime: Runtime) -> PreTrainedConfig:
    policy = PreTrainedConfig.from_pretrained(checkpoint, revision=revision)
    policy.pretrained_path = Path(checkpoint)
    policy.pretrained_revision = revision
    policy.device = runtime.device
    policy.use_amp = runtime.device == "cuda"
    policy.push_to_hub = False
    return policy


def load_latent_policy(
    checkpoint: str | Path,
    revision: str | None,
    runtime: Runtime,
    phase: str,
) -> LatentSmolVLAConfig:
    source = PreTrainedConfig.from_pretrained(checkpoint, revision=revision)
    values = {
        field.name: getattr(source, field.name)
        for field in dataclasses.fields(LatentSmolVLAConfig)
        if field.init and hasattr(source, field.name)
    }
    values.update(
        phase=phase,
        device=runtime.device,
        use_amp=runtime.device == "cuda",
        push_to_hub=False,
        pretrained_path=Path(checkpoint),
        pretrained_revision=revision,
        freeze_vision_encoder=True,
        chunk_size=50,
        n_action_steps=50,
        vision_encode_batch_size=16 if runtime.device == "mps" else 64,
    )
    return LatentSmolVLAConfig(**values)


def prepare_training(
    name: str,
    policy_config: PreTrainedConfig,
    data_config: DatasetConfig,
    steps: int,
    seed: int,
    runtime: Runtime,
) -> TrainingSetup:
    os.environ["MUJOCO_GL"] = "cgl" if runtime.device == "mps" else "egl"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    config = TrainPipelineConfig(
        dataset=data_config,
        policy=policy_config,
        output_dir=Path("outputs") / name,
        job_name=name,
        steps=steps,
        batch_size=runtime.batch_size,
        num_workers=runtime.workers,
        seed=seed,
        rename_map=RENAME,
        use_policy_training_preset=False,
    )
    data = make_dataset(config)
    policy = make_policy(policy_config, ds_meta=data.meta, rename_map=RENAME)
    stats = rename_stats(data.meta.stats, RENAME)
    source = str(policy_config.pretrained_path) if policy_config.pretrained_path else None
    preprocessor, postprocessor = make_pre_post_processors(
        policy_config,
        pretrained_path=source,
        pretrained_revision=policy_config.pretrained_revision,
        dataset_stats=stats,
        preprocessor_overrides={
            "device_processor": {"device": runtime.device},
            "normalizer_processor": {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
                "stats": stats,
            },
            "rename_observations_processor": {"rename_map": RENAME},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
                "stats": stats,
            }
        },
    )
    return TrainingSetup(config, data, policy, preprocessor, postprocessor)


def train(
    setup: TrainingSetup,
    runtime: Runtime,
    *,
    lr: float,
    warmup_steps: int,
    final_lr: float,
    grad_clip: float = 10.0,
) -> Path:
    output = setup.config.output_dir
    if output.exists():
        raise FileExistsError(f"Run output already exists: {output}")

    _seed_everything(setup.config.seed)
    trainable = [parameter for parameter in setup.policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("The method left no trainable parameters.")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_scale(step, setup.config.steps, warmup_steps, final_lr / lr),
    )
    loader = _loader(setup, runtime)
    observer = TrainingObserver(
        setup.config.job_name,
        {
            "runtime": runtime,
            "dataset": setup.config.dataset,
            "policy": setup.policy.config,
            "training": {
                "steps": setup.config.steps,
                "seed": setup.config.seed,
                "optimizer": "AdamW",
                "lr": lr,
                "final_lr": final_lr,
                "warmup_steps": warmup_steps,
                "grad_clip": grad_clip,
                "trainable_parameters": sum(p.numel() for p in trainable),
                "total_parameters": sum(p.numel() for p in setup.policy.parameters()),
            },
        },
        runtime.device,
    )

    try:
        for step, batch in enumerate(loader, 1):
            started = time.perf_counter()
            setup.policy.train()
            batch = setup.preprocessor(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16, enabled=runtime.device == "cuda"):
                loss, details = setup.policy(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            effective_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            elapsed = time.perf_counter() - started
            observer.log(
                step,
                {
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                    "lr": effective_lr,
                    "next_lr": optimizer.param_groups[0]["lr"],
                    "seconds": elapsed,
                    "samples_per_second": runtime.batch_size / elapsed,
                    **(details or {}),
                },
            )
        checkpoint = _save(setup)
        observer.finish(checkpoint)
        return checkpoint
    except Exception as error:
        observer.fail(error)
        raise


def _loader(setup: TrainingSetup, runtime: Runtime) -> DataLoader:
    generator = torch.Generator().manual_seed(setup.config.seed)
    sampler = RandomSampler(
        setup.dataset,
        replacement=True,
        num_samples=setup.config.steps * runtime.batch_size,
        generator=generator,
    )
    return DataLoader(
        setup.dataset,
        batch_size=runtime.batch_size,
        sampler=sampler,
        num_workers=runtime.workers,
        collate_fn=lerobot_collate_fn,
        pin_memory=runtime.device == "cuda",
        persistent_workers=runtime.workers > 0,
        multiprocessing_context="spawn" if runtime.workers > 0 else None,
    )


def _lr_scale(step: int, steps: int, warmup: int, final_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, steps - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return final_ratio + (1 - final_ratio) * cosine


def _save(setup: TrainingSetup) -> Path:
    checkpoint = setup.config.output_dir / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    setup.policy.save_pretrained(checkpoint)
    setup.policy.config.save_pretrained(checkpoint)
    setup.preprocessor.save_pretrained(checkpoint)
    setup.postprocessor.save_pretrained(checkpoint)
    (checkpoint / "train_config.json").write_text(
        json.dumps(plain(setup.config.to_dict()), indent=2, ensure_ascii=False) + "\n"
    )
    return checkpoint


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
