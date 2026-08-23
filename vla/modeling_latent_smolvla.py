from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.common.vla_utils import pad_vector, resize_with_pad
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from vla.configuration_latent_smolvla import LatentSmolVLAConfig


class TransitionHead(nn.Module):
    def __init__(self, hidden: int, steps: int, views: int):
        super().__init__()
        self.step_embedding = nn.Parameter(torch.zeros(steps, hidden))
        self.view_embedding = nn.Parameter(torch.zeros(views, hidden))
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, current: Tensor, language: Tensor) -> Tensor:
        hidden = (
            current[:, None]
            + language[:, None, None, None]
            + self.step_embedding[None, :, None, None]
            + self.view_embedding[None, None, :, None]
        )
        return self.predictor(hidden)


class LatentSmolVLAPolicy(SmolVLAPolicy):
    config_class = LatentSmolVLAConfig
    name = "latent_smolvla"

    def __init__(self, config: LatentSmolVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        hidden = self.model.vlm_with_expert.config.text_config.hidden_size
        self.transition_head = TransitionHead(hidden, config.chunk_size, config.camera_count)
        self.action_decoder = nn.Linear(config.camera_count * hidden, config.max_action_dim)
        self._set_trainable_phase()

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def _set_trainable_phase(self):
        self.requires_grad_(False)
        modules = (self.transition_head,) if self.config.phase == "transition" else (self.action_decoder,)
        for module in modules:
            module.requires_grad_(True)

    def _visual_features(self, batch: dict[str, Tensor]) -> Tensor:
        keys = [key for key in self.config.image_features if key in batch]
        if len(keys) != self.config.camera_count:
            raise ValueError(f"Expected {self.config.camera_count} cameras, found {keys}.")

        views = []
        for key in keys:
            frames = batch[key]
            if frames.ndim == 4:
                frames = frames[:, None]
            batch_size, time = frames.shape[:2]
            flat = frames.flatten(0, 1)
            width, height = self.config.resize_imgs_with_padding
            flat = resize_with_pad(flat, height, width, pad_value=0) * 2.0 - 1.0
            encoded = []
            with torch.no_grad():
                for part in flat.split(self.config.vision_encode_batch_size):
                    encoded.append(self.model.vlm_with_expert.embed_image(part))
            views.append(torch.cat(encoded).reshape(batch_size, time, *encoded[0].shape[1:]))
        return torch.stack(views, dim=2)

    def _language_feature(self, batch: dict[str, Tensor]) -> Tensor:
        tokens = batch[OBS_LANGUAGE_TOKENS]
        mask = batch[OBS_LANGUAGE_ATTENTION_MASK].to(tokens.device)
        with torch.no_grad():
            embeddings = self.model.vlm_with_expert.embed_language_tokens(tokens)
        weights = mask.unsqueeze(-1).to(embeddings.dtype)
        return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def _predict_transitions(self, visual: Tensor, language: Tensor) -> Tensor:
        return self.transition_head(visual[:, 0], language)

    def _transition_loss(self, batch: dict[str, Tensor]):
        visual = self._visual_features(batch)
        predicted = self._predict_transitions(visual, self._language_feature(batch))
        target = (visual[:, 1:] - visual[:, :-1]).to(predicted.dtype)
        keys = [key for key in self.config.image_features if key in batch]
        padding = batch.get(f"{keys[0]}_is_pad")
        valid = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        if padding is not None:
            valid = ~(padding[:, 1:] | padding[:, :-1])
        transition_mask = valid[:, :, None, None, None]
        loss = (((predicted - target) ** 2) * transition_mask).sum() / (
            valid.sum() * target.shape[2] * target.shape[3] * target.shape[4]
        ).clamp_min(1)
        token_mask = valid[:, :, None]
        cosine_values = F.cosine_similarity(predicted.flatten(3), target.flatten(3), dim=-1)
        cosine = (cosine_values * token_mask).sum() / (
            valid.sum() * target.shape[2]
        ).clamp_min(1)
        norm_count = (valid.sum() * target.shape[2] * target.shape[3]).clamp_min(1)
        return loss, {
            "transition_loss": loss.item(),
            "transition_cosine": cosine.item(),
            "target_delta_norm": (
                (target.norm(dim=-1) * valid[:, :, None, None]).sum() / norm_count
            ).item(),
            "predicted_delta_norm": (
                (predicted.norm(dim=-1) * valid[:, :, None, None]).sum() / norm_count
            ).item(),
        }

    def _action_loss(self, batch: dict[str, Tensor]):
        with torch.no_grad():
            visual = self._visual_features(batch)
            transitions = (visual[:, 1:] - visual[:, :-1]).to(self.action_decoder.weight.dtype)
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        decoded = self.action_decoder(transitions.mean(dim=3).flatten(2))
        losses = F.mse_loss(decoded, actions, reduction="none")
        padding = batch.get("action_is_pad")
        if padding is not None:
            losses = losses * (~padding).unsqueeze(-1)
            count = ((~padding).sum() * losses.shape[-1]).clamp_min(1)
            loss = losses.sum() / count
        else:
            loss = losses.mean()
        return loss, {
            "action_loss": loss.item(),
            "decoded_action_norm": decoded.norm(dim=-1).mean().item(),
            "target_action_norm": actions.norm(dim=-1).mean().item(),
        }

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        if reduction != "mean":
            raise ValueError("Latent SmolVLA does not use sample weighting.")
        if self.config.phase == "transition":
            return self._transition_loss(batch)
        return self._action_loss(batch)

    def _get_action_chunk(self, batch: dict[str, Tensor], noise=None, **kwargs) -> Tensor:
        visual = self._visual_features(batch)
        predicted = self._predict_transitions(visual, self._language_feature(batch))
        actions = self.action_decoder(predicted.mean(dim=3).flatten(2))
        return actions[:, :, : self.config.action_feature.shape[0]]
