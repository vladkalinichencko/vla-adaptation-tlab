from lerobot.configs.default import DatasetConfig

def use_image_augmentations(config: DatasetConfig) -> DatasetConfig:
    config.image_transforms.enable = True
    return config
