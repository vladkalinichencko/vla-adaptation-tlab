from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_policy

from vla.data import RENAME, Source, metadata


source = Source("local/official_libero_90_v3", "local", Path("/private/tmp/official_libero_90_v3_test"))
for name in ("boundary_explicit_loop", "boundary_explicit_lora", "boundary_explicit_latent"):
    checkpoint = Path("outputs") / name / "checkpoints" / "last" / "pretrained_model"
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.device = "mps"
    policy = make_policy(config, ds_meta=metadata(source), rename_map=RENAME)
    print(name, type(policy).__name__, sum(parameter.numel() for parameter in policy.parameters()))
