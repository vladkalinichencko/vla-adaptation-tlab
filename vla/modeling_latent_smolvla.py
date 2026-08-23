from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.common.vla_utils import pad_vector, resize_with_pad
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from vla.configuration_latent_smolvla import LatentSmolVLAConfig


class InverseModel(nn.Module):
    def __init__(self, hidden: int, views: int, latent: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * views * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent),
        )

    def forward(self, current: Tensor, next_frame: Tensor) -> Tensor:
        pair = torch.cat((current.mean(dim=-2), next_frame.mean(dim=-2)), dim=-1)
        return self.network(pair.flatten(-2))


class ForwardModel(nn.Module):
    def __init__(self, hidden: int, views: int, latent: int):
        super().__init__()
        self.latent_projection = nn.Linear(latent, hidden)
        self.view_embedding = nn.Parameter(torch.zeros(views, hidden))
        self.network = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, current: Tensor, latent: Tensor) -> Tensor:
        hidden = current + self.latent_projection(latent)[:, :, None, None]
        hidden = hidden + self.view_embedding[None, None, :, None]
        return self.network(hidden)


class LatentPolicyHead(nn.Module):
    def __init__(self, hidden: int, views: int, steps: int, latent: int):
        super().__init__()
        self.steps = steps
        self.latent = latent
        self.network = nn.Sequential(
            nn.Linear((views + 1) * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, steps * latent),
        )

    def forward(self, current: Tensor, language: Tensor) -> Tensor:
        context = torch.cat((current.mean(dim=-2).flatten(1), language), dim=-1)
        return self.network(context).reshape(current.shape[0], self.steps, self.latent)


class LatentSmolVLAPolicy(SmolVLAPolicy):
    config_class = LatentSmolVLAConfig
    name = "latent_smolvla"

    def __init__(self, config: LatentSmolVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        hidden = self.model.vlm_with_expert.config.text_config.hidden_size
        self.inverse_model = InverseModel(hidden, config.camera_count, config.latent_dim)
        self.forward_model = ForwardModel(hidden, config.camera_count, config.latent_dim)
        self.latent_policy = LatentPolicyHead(
            hidden, config.camera_count, config.chunk_size, config.latent_dim
        )
        self.action_decoder = nn.Linear(config.latent_dim, config.max_action_dim)
        self._set_trainable_phase()

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def _set_trainable_phase(self):
        self.requires_grad_(False)
        modules = {
            "representation": (self.inverse_model, self.forward_model),
            "policy": (self.latent_policy,),
            "action": (self.action_decoder,),
        }[self.config.phase]
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

    def _valid_transitions(self, batch: dict[str, Tensor], steps: int) -> Tensor:
        keys = [key for key in self.config.image_features if key in batch]
        padding = batch.get(f"{keys[0]}_is_pad")
        if padding is None:
            return torch.ones(
                batch[ACTION].shape[0], steps, dtype=torch.bool, device=batch[ACTION].device
            )
        return ~(padding[:, 1 : steps + 1] | padding[:, :steps])

    def _representation_loss(self, batch: dict[str, Tensor]):
        visual = self._visual_features(batch)
        current, next_frame = visual[:, :-1], visual[:, 1:]
        latent = self.inverse_model(current, next_frame)
        predicted = self.forward_model(current, latent)
        target = (visual[:, 1:] - visual[:, :-1]).to(predicted.dtype)
        valid = self._valid_transitions(batch, target.shape[1])
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
        with torch.no_grad():
            zero = self.forward_model(current, torch.zeros_like(latent))
            zero_loss = (((zero - target) ** 2) * transition_mask).sum() / (
                valid.sum() * target.shape[2] * target.shape[3] * target.shape[4]
            ).clamp_min(1)
        return loss, {
            "representation_loss": loss.item(),
            "transition_cosine": cosine.item(),
            "target_delta_norm": (
                (target.norm(dim=-1) * valid[:, :, None, None]).sum() / norm_count
            ).item(),
            "predicted_delta_norm": (
                (predicted.norm(dim=-1) * valid[:, :, None, None]).sum() / norm_count
            ).item(),
            "latent_norm": (latent.norm(dim=-1) * valid).sum().div(valid.sum().clamp_min(1)).item(),
            "zero_latent_loss": zero_loss.item(),
        }

    def _policy_loss(self, batch: dict[str, Tensor]):
        with torch.no_grad():
            visual = self._visual_features(batch)
            target = self.inverse_model(visual[:, :-1], visual[:, 1:])
            language = self._language_feature(batch)
        predicted = self.latent_policy(visual[:, 0], language)
        valid = self._valid_transitions(batch, predicted.shape[1])
        losses = ((predicted - target) ** 2) * valid.unsqueeze(-1)
        loss = losses.sum() / (valid.sum() * predicted.shape[-1]).clamp_min(1)
        cosine = F.cosine_similarity(predicted, target, dim=-1)
        return loss, {
            "latent_policy_loss": loss.item(),
            "latent_policy_cosine": (cosine * valid).sum().div(valid.sum().clamp_min(1)).item(),
            "target_latent_norm": (target.norm(dim=-1) * valid).sum().div(valid.sum().clamp_min(1)).item(),
            "predicted_latent_norm": (predicted.norm(dim=-1) * valid).sum().div(valid.sum().clamp_min(1)).item(),
        }

    def _action_loss(self, batch: dict[str, Tensor]):
        with torch.no_grad():
            visual = self._visual_features(batch)
            latent = self.inverse_model(visual[:, :-1], visual[:, 1:]).to(
                self.action_decoder.weight.dtype
            )
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        decoded = self.action_decoder(latent)
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
        return {
            "representation": self._representation_loss,
            "policy": self._policy_loss,
            "action": self._action_loss,
        }[self.config.phase](batch)

    def _get_action_chunk(self, batch: dict[str, Tensor], noise=None, **kwargs) -> Tensor:
        visual = self._visual_features(batch)
        latent = self.latent_policy(visual[:, 0], self._language_feature(batch))
        actions = self.action_decoder(latent)
        return actions[:, :, : self.config.action_feature.shape[0]]
