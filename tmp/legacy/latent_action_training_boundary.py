from pathlib import Path

from vla.behavior import action_snapshot
from vla.data import Source, dataset
from vla.runtime import Runtime
from vla.training import load_latent_policy, train_policy


runtime = Runtime("mps", 1, 0, "no", 1, (0,), (5,), (0,))
source = Source(
    "local/official_libero_90_v3",
    "local",
    Path("/private/tmp/official_libero_90_v3_test"),
)
transition = Path("outputs/latent_boundary_smoke/checkpoints/last/pretrained_model")
policy = load_latent_policy(transition, None, runtime, "action")
checkpoint = train_policy("latent_action_boundary_smoke", policy, dataset(source, [0]), 1, 0, runtime)
print(action_snapshot("latent_action_boundary_smoke", checkpoint, source, [0], runtime))
