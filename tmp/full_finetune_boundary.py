from pathlib import Path

from vla.data import BASE_POLICY, BASE_POLICY_REVISION, Source, dataset
from vla.methods import use_full_finetune
from vla.runtime import Runtime
from vla.training import load_policy, prepare_training


runtime = Runtime("mps", 1, 0, "no", 1, (0,), (5,), (0,))
source = Source("local/official_libero_90_v3", "local", Path("/private/tmp/official_libero_90_v3_test"))
config = use_full_finetune(load_policy(BASE_POLICY, BASE_POLICY_REVISION, runtime))
setup = prepare_training("boundary_full_finetune", config, dataset(source, [0]), 1, 0, runtime)
print(sum(parameter.numel() for parameter in setup.policy.parameters() if parameter.requires_grad))
