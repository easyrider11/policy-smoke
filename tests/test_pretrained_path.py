"""Check 4: PreTrainedConfig.from_pretrained(path) should remember where it came from (lerobot#4647).

On lerobot 0.6.1 it does not: ``pretrained_path`` stays ``None``. Code that then calls
``make_pre_post_processors(cfg, pretrained_path=cfg.pretrained_path)`` builds fresh
processors with no stats, and normalization becomes a silent identity.

Both tests are ``xfail(strict=True)``: when upstream fixes this, they start passing,
the suite turns red, and the marker should be removed.
"""

import pytest
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors

from policy_smoke.builders import make_fake_stats
from policy_smoke.checks import missing_normalization_stats

BUG = "lerobot#4647: PreTrainedConfig.from_pretrained leaves pretrained_path=None"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=BUG)
def test_config_from_pretrained_sets_pretrained_path(checkpoint):
    config = PreTrainedConfig.from_pretrained(checkpoint)
    assert config.pretrained_path is not None
    assert str(config.pretrained_path) == str(checkpoint)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=BUG)
def test_processors_from_loaded_config_keep_stats(policy, checkpoint):
    pre, post = make_pre_post_processors(policy.config, dataset_stats=make_fake_stats(policy.config))
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)

    config = PreTrainedConfig.from_pretrained(checkpoint)
    loaded_pre, loaded_post = make_pre_post_processors(config, pretrained_path=config.pretrained_path)

    assert missing_normalization_stats(loaded_pre) == []
    assert missing_normalization_stats(loaded_post) == []
