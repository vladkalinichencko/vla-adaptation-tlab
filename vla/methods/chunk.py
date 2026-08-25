from lerobot.configs.policies import PreTrainedConfig

def use_action_chunk(config: PreTrainedConfig, size: int) -> PreTrainedConfig:
    config.chunk_size = size
    config.n_action_steps = size
    return config
