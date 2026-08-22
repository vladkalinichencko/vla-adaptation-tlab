from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("latent_smolvla")
@dataclass
class LatentSmolVLAConfig(SmolVLAConfig):
    phase: str = "transition"
    camera_count: int = 2
    vision_encode_batch_size: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.phase not in ("transition", "action"):
            raise ValueError(f"Unknown latent training phase: {self.phase}")
        if self.chunk_size != 50 or self.n_action_steps != 50:
            raise ValueError("Latent SmolVLA uses the original 50-action chunk.")
        if not self.freeze_vision_encoder:
            raise ValueError("The latent target requires a frozen vision encoder.")

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size + 1))
