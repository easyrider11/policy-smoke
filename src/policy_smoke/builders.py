"""Tiny, randomly initialised LeRobot policies that build offline on CPU.

Every config here avoids network access: pretrained vision backbones are
disabled and no policy that needs a Hub-hosted VLM/text encoder is included.
Dims are kept small so a full save/load/infer cycle takes well under a second.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.pretrained import PreTrainedPolicy

STATE_DIM = 6
ACTION_DIM = 6
IMAGE_SHAPE = (3, 64, 64)

STATE = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))}
IMAGE = {"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE)}
ACTION = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}


def _act() -> dict[str, Any]:
    return dict(
        input_features={**STATE, **IMAGE},
        chunk_size=8,
        n_action_steps=4,
        pretrained_backbone_weights=None,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
    )


def _diffusion() -> dict[str, Any]:
    return dict(
        input_features={**STATE, **IMAGE},
        n_obs_steps=2,
        horizon=8,
        n_action_steps=4,
        pretrained_backbone_weights=None,
        crop_shape=None,
        down_dims=(16, 32),
        n_groups=4,
        diffusion_step_embed_dim=16,
        spatial_softmax_num_keypoints=8,
        num_train_timesteps=10,
    )


def _vqbet() -> dict[str, Any]:
    return dict(
        input_features={**STATE, **IMAGE},
        n_obs_steps=2,
        n_action_pred_token=2,
        action_chunk_size=2,
        pretrained_backbone_weights=None,
        crop_shape=None,
        spatial_softmax_num_keypoints=8,
        vqvae_embedding_dim=16,
        vqvae_enc_hidden_dim=16,
        gpt_block_size=20,
        gpt_input_dim=32,
        gpt_output_dim=32,
        gpt_n_layer=1,
        gpt_n_head=2,
        gpt_hidden_dim=32,
    )


def _tdmpc() -> dict[str, Any]:
    return dict(
        input_features={**STATE},
        horizon=2,
        n_action_steps=2,
        n_action_repeats=1,
        state_encoder_hidden_dim=32,
        latent_dim=16,
        mlp_dim=32,
        q_ensemble_size=2,
        n_gaussian_samples=16,
        n_pi_samples=4,
        n_elites=4,
        cem_iterations=2,
    )


def _gaussian_actor() -> dict[str, Any]:
    return dict(input_features={**STATE}, state_encoder_hidden_dim=32, latent_dim=32)


TINY_CONFIGS = {
    "act": _act,
    "diffusion": _diffusion,
    "vqbet": _vqbet,
    "tdmpc": _tdmpc,
    "gaussian_actor": _gaussian_actor,
}

# Policies registered in lerobot 0.6.1 that this suite does not build, and what
# happened when we tried with HF_HUB_OFFLINE=1 on CPU.
SKIPPED_POLICIES = {
    "smolvla": "constructor downloads the SmolVLM config/processor from huggingface.co",
    "multi_task_dit": "constructor downloads a CLIP text encoder from huggingface.co",
    "eo1, evo1, vla_jepa": "constructor downloads a VLM config/processor from huggingface.co",
    "groot, molmoact2": "constructor needs a Hub snapshot (fails offline)",
    "pi0, pi05, lingbot_va": "builds a multi-billion-parameter model; did not finish in 120 s on CPU",
    "pi0_fast, wall_x": "extra deps (scipy / peft) plus a Hub-hosted VLM",
    "xvla": "needs a full vision_config for its Florence-2 backbone",
    "fastwam": "hard-codes a 7-dim action space and a large video model",
}

OFFLINE_POLICIES = sorted(TINY_CONFIGS)


def make_tiny_config(name: str, **overrides: Any) -> PreTrainedConfig:
    """Return a small CPU config for policy type ``name``."""
    kwargs = {"output_features": dict(ACTION), "device": "cpu", **TINY_CONFIGS[name](), **overrides}
    return make_policy_config(name, **kwargs)


def make_tiny_policy(name: str, seed: int = 0, **overrides: Any) -> PreTrainedPolicy:
    """Build a randomly initialised policy in eval mode. Same seed gives same weights."""
    config = make_tiny_config(name, **overrides)
    torch.manual_seed(seed)
    policy = get_policy_class(name)(config)
    policy.eval()
    return policy


def make_observation(config: PreTrainedConfig, batch_size: int = 1, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random observation batch matching ``config.input_features``."""
    gen = torch.Generator().manual_seed(seed)
    return {
        key: torch.rand((batch_size, *ft.shape), generator=gen) for key, ft in config.input_features.items()
    }


def make_fake_stats(config: PreTrainedConfig) -> dict[str, dict[str, torch.Tensor]]:
    """Non-trivial dataset stats (not mean 0 / std 1) for every input and output feature."""
    stats: dict[str, dict[str, torch.Tensor]] = {}
    features = {**config.input_features, **config.output_features}
    for i, (key, ft) in enumerate(sorted(features.items())):
        shape = (ft.shape[0], 1, 1) if ft.type is FeatureType.VISUAL else ft.shape
        base = torch.linspace(0.1, 0.9, shape[0]).reshape(shape)
        stats[key] = {
            "mean": base + i,
            "std": torch.full(shape, 2.0 + i),
            "min": base - 3.0,
            "max": base + 3.0,
        }
    return stats


def expected_replan_period(config: PreTrainedConfig) -> int | None:
    """How many ``select_action`` calls one ``predict_action_chunk`` result should cover.

    ``None`` means the policy does not plan in chunks (e.g. gaussian_actor).
    """
    if config.type == "vqbet":
        return config.action_chunk_size
    if config.type == "gaussian_actor":
        return None
    return config.n_action_steps
