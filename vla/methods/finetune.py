from lerobot.configs.policies import PreTrainedConfig

def use_full_finetune(config: PreTrainedConfig) -> PreTrainedConfig:
    config.freeze_vision_encoder = False
    config.train_expert_only = False
    return config
