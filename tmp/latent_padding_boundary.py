from types import SimpleNamespace

import torch
from torch import nn

from vla.modeling_latent_smolvla import LatentSmolVLAPolicy


class DummyPolicy(LatentSmolVLAPolicy):
    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(image_features=("camera",))
        self.inverse_model = ZeroInverse()
        self.forward_model = ZeroForward()

    def _visual_features(self, batch):
        return batch["visual"]


class ZeroInverse(nn.Module):
    def forward(self, current, next_frame):
        return torch.zeros(*current.shape[:2], 1)


class ZeroForward(nn.Module):
    def forward(self, current, latent):
        return torch.zeros_like(current)


policy = DummyPolicy()
visual = torch.tensor([0.0, 1.0, 2.0, 2.0]).reshape(1, 4, 1, 1, 1)
loss, details = policy._representation_loss({
    "action": torch.empty(1, 3, 1),
    "camera": torch.empty(1),
    "camera_is_pad": torch.tensor([[False, False, False, True]]),
    "visual": visual,
})

assert loss.item() == 1.0
assert details["target_delta_norm"] == 1.0
assert details["predicted_delta_norm"] == 0.0
