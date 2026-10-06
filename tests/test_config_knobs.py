"""Config fields that are accepted and saved but have no effect (lerobot#4727)."""

import pytest
import torch
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import PolicyConfig

from policy_smoke.builders import make_observation, make_tiny_policy
from policy_smoke.checks import seeded_action


def _max_abs_action(use_tanh_squash: bool) -> float:
    policy = make_tiny_policy("gaussian_actor", policy_kwargs=PolicyConfig(use_tanh_squash=use_tanh_squash))
    with torch.no_grad():
        policy.actor.mean_layer.bias.fill_(5.0)  # push the mean far outside [-1, 1]
    return seeded_action(policy, make_observation(policy.config, batch_size=4)).abs().max().item()


def test_tanh_squash_on_bounds_actions():
    assert _max_abs_action(use_tanh_squash=True) <= 1.0


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="lerobot#4727: GaussianActor ignores use_tanh_squash=False"
)
def test_tanh_squash_off_leaves_actions_unbounded():
    assert _max_abs_action(use_tanh_squash=False) > 1.0
