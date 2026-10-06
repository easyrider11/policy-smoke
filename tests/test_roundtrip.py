"""Checks 1 and 2: save -> load round trip and seeded determinism."""

import torch
from lerobot.configs.policies import PreTrainedConfig

from policy_smoke.builders import make_observation
from policy_smoke.checks import config_differences, seeded_action, state_dict_differences


def test_state_dict_and_config_survive_roundtrip(policy, checkpoint):
    reloaded = type(policy).from_pretrained(checkpoint)

    assert state_dict_differences(policy.state_dict(), reloaded.state_dict()) == []
    assert config_differences(policy.config, reloaded.config) == []
    assert config_differences(policy.config, PreTrainedConfig.from_pretrained(checkpoint)) == []


def test_same_seed_same_action_before_and_after_roundtrip(policy, checkpoint):
    obs = make_observation(policy.config, batch_size=2)

    first = seeded_action(policy, obs, seed=123)
    again = seeded_action(policy, obs, seed=123)
    reloaded = seeded_action(type(policy).from_pretrained(checkpoint), obs, seed=123)

    torch.testing.assert_close(first, again, rtol=0, atol=0)
    torch.testing.assert_close(first, reloaded, rtol=0, atol=0)
