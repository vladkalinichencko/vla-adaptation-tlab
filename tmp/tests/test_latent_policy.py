import unittest

import torch

from vla.configuration_latent_smolvla import LatentSmolVLAConfig
from vla.modeling_latent_smolvla import ForwardModel, InverseModel, LatentPolicyHead


class LatentPolicyTest(unittest.TestCase):
    def test_representation_phase_loads_51_frames_for_50_actions(self):
        config = LatentSmolVLAConfig(device="cpu", phase="representation")
        self.assertEqual(config.observation_delta_indices, list(range(51)))
        self.assertEqual(config.action_delta_indices, list(range(50)))

    def test_action_phase_learns_decoder_from_real_transitions(self):
        config = LatentSmolVLAConfig(device="cpu", phase="action")
        self.assertEqual(config.observation_delta_indices, list(range(51)))
        self.assertEqual(config.action_delta_indices, list(range(50)))

    def test_lapo_modules_keep_time_and_camera_axes(self):
        current = torch.randn(3, 50, 2, 4, 8)
        next_frame = torch.randn_like(current)
        inverse = InverseModel(hidden=8, views=2, latent=4)
        forward = ForwardModel(hidden=8, views=2, latent=4)
        policy = LatentPolicyHead(hidden=8, views=2, steps=50, latent=4)

        latent = inverse(current, next_frame)
        self.assertEqual(latent.shape, (3, 50, 4))
        self.assertEqual(forward(current, latent).shape, current.shape)
        self.assertEqual(policy(current[:, 0], torch.randn(3, 8)).shape, latent.shape)


if __name__ == "__main__":
    unittest.main()
