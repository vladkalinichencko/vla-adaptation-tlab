from types import SimpleNamespace

import torch
from torch import nn

from vla.modeling_latent_smolvla import LatentSmolVLAPolicy


class DummyPolicy(LatentSmolVLAPolicy):
    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(image_features=("camera",))

    def _visual_features(self, batch):
        return batch["visual"]

    def _language_feature(self, batch):
        return torch.empty(1)

    def _predict_transitions(self, visual, language):
        return torch.zeros_like(visual[:, 1:])


policy = DummyPolicy()
visual = torch.tensor([0.0, 1.0, 2.0, 2.0]).reshape(1, 4, 1, 1, 1)
loss, details = policy._transition_loss({
    "camera": torch.empty(1),
    "camera_is_pad": torch.tensor([[False, False, False, True]]),
    "visual": visual,
})

assert loss.item() == 1.0
assert details["target_delta_norm"] == 1.0
assert details["predicted_delta_norm"] == 0.0
