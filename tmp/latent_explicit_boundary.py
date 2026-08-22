from pathlib import Path

from vla.data import BASE_POLICY, BASE_POLICY_REVISION, Source, dataset
from vla.runtime import Runtime
from vla.training import load_latent_policy, prepare_training, train


runtime = Runtime("mps", 1, 0, "no", 1, (0,), (5,), (0,))
source = Source("local/official_libero_90_v3", "local", Path("/private/tmp/official_libero_90_v3_test"))
config = load_latent_policy(BASE_POLICY, BASE_POLICY_REVISION, runtime, "transition")
setup = prepare_training("boundary_explicit_latent", config, dataset(source, [0]), 1, 0, runtime)
print(train(setup, runtime, lr=1e-4, warmup_steps=0, final_lr=2.5e-6))
