import unittest

import torch

from vla.configuration_latent_smolvla import LatentSmolVLAConfig
from vla.modeling_latent_smolvla import TransitionHead


class LatentPolicyTest(unittest.TestCase):
    def test_transition_phase_loads_51_frames_for_50_actions(self):
        config = LatentSmolVLAConfig(device="cpu", phase="transition")
        self.assertEqual(config.observation_delta_indices, list(range(51)))
        self.assertEqual(config.action_delta_indices, list(range(50)))

    def test_action_phase_learns_decoder_from_real_transitions(self):
        config = LatentSmolVLAConfig(device="cpu", phase="action")
        self.assertEqual(config.observation_delta_indices, list(range(51)))
        self.assertEqual(config.action_delta_indices, list(range(50)))

    def test_transition_head_keeps_time_and_camera_axes(self):
        head = TransitionHead(hidden=8, steps=50, views=2)
        predicted = head(torch.randn(3, 2, 4, 8), torch.randn(3, 8))
        self.assertEqual(predicted.shape, (3, 50, 2, 4, 8))
        predicted.square().mean().backward()
        self.assertIsNotNone(head.predictor[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
