"""Check 6: action shapes and chunk/replan cadence match the config."""

import pytest

from policy_smoke.builders import ACTION_DIM, expected_replan_period, make_observation
from policy_smoke.checks import action_shape_problems, trace_chunks

BATCH = 3  # differs from every horizon/chunk size used, so axis mix-ups show up


def test_select_action_shape(policy):
    assert policy.config.action_feature.shape == (ACTION_DIM,)
    assert action_shape_problems(policy, make_observation(policy.config, batch_size=BATCH)) == []


def test_replans_every_n_action_steps(policy):
    period = expected_replan_period(policy.config)
    if period is None:
        pytest.skip(f"{policy.config.type} does not predict action chunks")

    actions, chunk_calls = trace_chunks(policy, make_observation(policy.config, batch_size=BATCH), 3 * period)

    assert [step for step, _ in chunk_calls] == [0, period, 2 * period]
    assert all(tuple(a.shape) == (BATCH, ACTION_DIM) for a in actions)


def test_chunk_is_batch_major(policy, request):
    period = expected_replan_period(policy.config)
    if period is None:
        pytest.skip(f"{policy.config.type} does not predict action chunks")
    if policy.config.type == "tdmpc":
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="tdmpc predict_action_chunk returns (horizon, batch, action_dim), not (batch, ...)",
            )
        )

    _, chunk_calls = trace_chunks(policy, make_observation(policy.config, batch_size=BATCH), 1)
    batch, length, dim = chunk_calls[0][1]

    assert (batch, dim) == (BATCH, ACTION_DIM)
    assert length >= period
