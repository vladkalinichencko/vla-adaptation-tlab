from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import PreTrainedPolicy


def apply_lora(policy: PreTrainedPolicy, rank: int = 32) -> PreTrainedPolicy:
    return policy.wrap_with_peft(
        peft_cli_overrides={"method_type": "LORA", "r": rank, "lora_alpha": rank}
    )


def use_full_finetune(config: PreTrainedConfig) -> PreTrainedConfig:
    config.freeze_vision_encoder = False
    config.train_expert_only = False
    return config


def use_action_chunk(config: PreTrainedConfig, size: int) -> PreTrainedConfig:
    config.chunk_size = size
    config.n_action_steps = size
    return config


def use_image_augmentations(config: DatasetConfig) -> DatasetConfig:
    config.image_transforms.enable = True
    return config
