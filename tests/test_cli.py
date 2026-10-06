"""`policy-smoke check <dir>` on good and broken checkpoints."""

import pytest
from lerobot.policies.factory import make_pre_post_processors

from policy_smoke.builders import make_fake_stats, make_tiny_policy
from policy_smoke.cli import main

from .conftest import rewrite_weights


@pytest.fixture
def act_checkpoint(tmp_path):
    policy = make_tiny_policy("act")
    pre, post = make_pre_post_processors(policy.config, dataset_stats=make_fake_stats(policy.config))
    policy.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    return tmp_path


def test_good_checkpoint_passes(act_checkpoint, capsys):
    assert main(["check", str(act_checkpoint)]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert out.count("PASS") == 5


def test_dropped_key_fails(act_checkpoint, capsys):
    rewrite_weights(act_checkpoint, drop="model.action_head.weight")
    assert main(["check", str(act_checkpoint)]) == 1
    assert "missing key model.action_head.weight" in capsys.readouterr().out


def test_empty_config_fails(act_checkpoint, capsys):
    """The lerobot#4649 output: empty config.json."""
    (act_checkpoint / "config.json").write_text("")
    assert main(["check", str(act_checkpoint)]) == 1
    assert "config.json is empty" in capsys.readouterr().out


def test_processors_without_stats_fail(tmp_path, capsys):
    policy = make_tiny_policy("act")
    pre, post = make_pre_post_processors(policy.config, dataset_stats=None)
    policy.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)

    assert main(["check", str(tmp_path)]) == 1
    assert "preprocessor has no stats for observation.state" in capsys.readouterr().out
