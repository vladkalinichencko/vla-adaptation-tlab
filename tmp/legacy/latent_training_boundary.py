from pathlib import Path

from vla.behavior import transition_snapshot
from vla.data import BASE_POLICY, BASE_POLICY_REVISION, Source, dataset
from vla.runtime import Runtime
from vla.training import load_latent_policy, train_policy


runtime = Runtime("mps", 1, 0, "no", 1, (0,), (5,), (0,))
source = Source(
    "local/official_libero_90_v3",
    "local",
    Path("/private/tmp/official_libero_90_v3_test"),
)
policy = load_latent_policy(BASE_POLICY, BASE_POLICY_REVISION, runtime, "transition")
checkpoint = train_policy("latent_boundary_smoke", policy, dataset(source, [0]), 1, 0, runtime)
print(transition_snapshot("latent_boundary_smoke", checkpoint, source, [0], runtime))
