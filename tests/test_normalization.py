"""Check 5: normalization stats survive save/load, and missing stats are detectable (lerobot#4647)."""

import torch
from lerobot.policies.factory import make_pre_post_processors

from policy_smoke.builders import ACTION_DIM, make_fake_stats, make_observation
from policy_smoke.checks import missing_normalization_stats, normalizer_steps


def _sample(config) -> dict[str, torch.Tensor]:
    sample = make_observation(config, batch_size=1, seed=7)
    sample["action"] = torch.rand(1, ACTION_DIM)
    return sample


def test_stats_and_outputs_survive_processor_roundtrip(policy, tmp_path):
    config = policy.config
    pre, post = make_pre_post_processors(config, dataset_stats=make_fake_stats(config))
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)

    pre2, post2 = make_pre_post_processors(config, pretrained_path=str(tmp_path))

    for a, b in zip(
        normalizer_steps(pre) + normalizer_steps(post),
        normalizer_steps(pre2) + normalizer_steps(post2),
        strict=True,
    ):
        assert a._tensor_stats.keys() == b._tensor_stats.keys()
        for key in a._tensor_stats:
            for name, tensor in a._tensor_stats[key].items():
                torch.testing.assert_close(tensor, b._tensor_stats[key][name], rtol=0, atol=0)
    assert missing_normalization_stats(pre2) == []
    assert missing_normalization_stats(post2) == []

    sample = _sample(config)
    out, out2 = pre(dict(sample)), pre2(dict(sample))
    for key in sample:
        torch.testing.assert_close(out[key], out2[key], rtol=0, atol=0)
    action = torch.rand(1, ACTION_DIM)
    torch.testing.assert_close(post(action), post2(action), rtol=0, atol=0)


def test_processors_without_stats_are_detectable(policy):
    """No stats -> lerobot passes data through unchanged. The helper must flag it."""
    config = policy.config
    pre, post = make_pre_post_processors(config, dataset_stats=None)

    missing = missing_normalization_stats(pre)
    assert "action" in missing
    assert missing_normalization_stats(post) == ["action"]

    sample = _sample(config)
    out = pre(dict(sample))
    for key in missing:
        torch.testing.assert_close(out[key], sample[key], rtol=0, atol=0)  # silent identity
