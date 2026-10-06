"""Check 3: missing / renamed weight keys must be detectable (lerobot#4711 failure class)."""

import pytest
import torch

from policy_smoke.checks import load_strict_report, state_dict_differences

from .conftest import rewrite_weights


def _some_key(policy) -> str:
    # Last randomly initialised weight matrix (some biases start at zero, so a fresh
    # init would match them by accident).
    return [name for name, p in policy.named_parameters() if p.dim() > 1][-1]


def test_clean_checkpoint_reports_no_mismatch(policy, checkpoint):
    assert load_strict_report(type(policy), checkpoint).clean


def test_dropped_key_loads_silently_by_default(policy, checkpoint):
    """Documents the failure class: lerobot's default from_pretrained does not raise."""
    key = _some_key(policy)
    rewrite_weights(checkpoint, drop=key)

    torch.manual_seed(999)  # different init than the saved policy
    reloaded = type(policy).from_pretrained(checkpoint)  # strict=False by default

    assert f"{key}: values differ" in state_dict_differences(policy.state_dict(), reloaded.state_dict())


def test_helper_catches_dropped_key(policy, checkpoint):
    key = _some_key(policy)
    rewrite_weights(checkpoint, drop=key)

    report = load_strict_report(type(policy), checkpoint)

    assert report.missing == [key]
    assert report.unexpected == []


def test_helper_catches_renamed_key(policy, checkpoint):
    key = _some_key(policy)
    rewrite_weights(checkpoint, rename=(key, "legacy." + key))

    report = load_strict_report(type(policy), checkpoint)

    assert report.missing == [key]
    assert report.unexpected == ["legacy." + key]


def test_strict_true_raises_on_dropped_key(policy, checkpoint):
    rewrite_weights(checkpoint, drop=_some_key(policy))
    with pytest.raises(RuntimeError, match="Missing key"):
        type(policy).from_pretrained(checkpoint, strict=True)
