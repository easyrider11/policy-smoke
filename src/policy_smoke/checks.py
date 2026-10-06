"""Pure check functions. Each returns data (lists of problems, reports) and never prints."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from safetensors.torch import load_model

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
PREPROCESSOR_FILE = "policy_preprocessor.json"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class LoadReport:
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.missing and not self.unexpected


# --- files ----------------------------------------------------------------------------


def checkpoint_file_problems(path: str | Path) -> list[str]:
    """Problems with the files of a saved policy dir (lerobot#4649: empty config, no weights)."""
    path = Path(path)
    problems = []
    config_file = path / CONFIG_FILE
    if not config_file.is_file():
        problems.append(f"{CONFIG_FILE} missing")
    elif config_file.stat().st_size == 0:
        problems.append(f"{CONFIG_FILE} is empty")
    else:
        try:
            data = json.loads(config_file.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{CONFIG_FILE} is not valid JSON: {e}")
        else:
            if not data:
                problems.append(f"{CONFIG_FILE} is an empty object")
            elif "type" not in data:
                problems.append(f"{CONFIG_FILE} has no 'type' field")
    weights = path / WEIGHTS_FILE
    if not weights.is_file():
        problems.append(f"{WEIGHTS_FILE} missing")
    elif weights.stat().st_size == 0:
        problems.append(f"{WEIGHTS_FILE} is empty")
    return problems


# --- loading --------------------------------------------------------------------------


def load_config_offline(path: str | Path) -> PreTrainedConfig:
    """Load a policy config for CPU use without any downloads.

    Pretrained torchvision backbones are switched off: their weights are in
    model.safetensors anyway, and fetching them would need the network.
    """
    config = PreTrainedConfig.from_pretrained(path)
    config.device = "cpu"
    if getattr(config, "pretrained_backbone_weights", None) is not None:
        config.pretrained_backbone_weights = None
    return config


def load_strict_report(
    policy_cls: type[PreTrainedPolicy], path: str | Path, config: PreTrainedConfig | None = None
) -> LoadReport:
    """Load ``path/model.safetensors`` into a fresh ``policy_cls`` and report key mismatches.

    lerobot's ``from_pretrained`` defaults to ``strict=False`` and only logs mismatches,
    so a checkpoint with dropped or renamed keys loads "fine" with random weights in
    those slots (lerobot#4711). This helper returns the mismatches instead.
    """
    config = config or load_config_offline(path)
    policy = policy_cls(config)
    missing, unexpected = load_model(policy, str(Path(path) / WEIGHTS_FILE), strict=False, device="cpu")
    return LoadReport(missing=sorted(missing), unexpected=sorted(unexpected))


def load_policy_strict(path: str | Path) -> PreTrainedPolicy:
    """Load a local checkpoint on CPU, raising if any weight key is missing or unexpected."""
    config = load_config_offline(path)
    policy_cls = get_policy_class(config.type)
    report = load_strict_report(policy_cls, path, config)
    if not report.clean:
        raise RuntimeError(f"key mismatch: missing={report.missing} unexpected={report.unexpected}")
    with contextlib.redirect_stdout(io.StringIO()):  # lerobot prints "Loading weights from ..."
        policy = policy_cls.from_pretrained(path, config=config, strict=True)
    return policy.eval()


# --- comparisons ----------------------------------------------------------------------


def state_dict_differences(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[str]:
    """Keys that are missing on one side or whose tensors differ (shape, dtype or values)."""
    diffs = [f"only in first: {k}" for k in sorted(a.keys() - b.keys())]
    diffs += [f"only in second: {k}" for k in sorted(b.keys() - a.keys())]
    for key in sorted(a.keys() & b.keys()):
        ta, tb = a[key], b[key]
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            diffs.append(f"{key}: {tuple(ta.shape)}/{ta.dtype} vs {tuple(tb.shape)}/{tb.dtype}")
        elif not torch.equal(ta.cpu(), tb.cpu()):
            diffs.append(f"{key}: values differ")
    return diffs


def config_differences(a: PreTrainedConfig, b: PreTrainedConfig) -> list[str]:
    """Top-level config fields whose serialized values differ."""
    ea = draccus.encode(a, PreTrainedConfig)
    eb = draccus.encode(b, PreTrainedConfig)
    return [k for k in sorted(ea.keys() | eb.keys()) if ea.get(k) != eb.get(k)]


def roundtrip_differences(policy: PreTrainedPolicy) -> list[str]:
    """Save ``policy``, reload it, and list state-dict and config differences."""
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        policy.save_pretrained(tmp)
        reloaded = type(policy).from_pretrained(tmp, config=load_config_offline(tmp), strict=True)
    diffs = state_dict_differences(policy.state_dict(), reloaded.state_dict())
    diffs += [f"config field: {k}" for k in config_differences(policy.config, reloaded.config)]
    return diffs


# --- inference ------------------------------------------------------------------------


@torch.no_grad()
def seeded_action(
    policy: PreTrainedPolicy, observation: dict[str, torch.Tensor], seed: int = 0
) -> torch.Tensor:
    """One ``select_action`` call from a fresh queue with a fixed torch seed."""
    policy.eval()
    policy.reset()
    torch.manual_seed(seed)
    return policy.select_action(dict(observation))


@torch.no_grad()
def trace_chunks(policy: PreTrainedPolicy, observation: dict[str, torch.Tensor], n_steps: int):
    """Call ``select_action`` ``n_steps`` times and record when a new chunk was predicted.

    Returns ``(actions, chunk_calls)`` where ``chunk_calls`` is a list of
    ``(step_index, chunk_shape)`` for every ``predict_action_chunk`` call.
    """
    chunk_calls: list[tuple[int, tuple[int, ...]]] = []
    actions: list[torch.Tensor] = []
    original = policy.predict_action_chunk

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        chunk_calls.append((len(actions), tuple(out.shape)))
        return out

    policy.eval()
    policy.reset()
    policy.predict_action_chunk = wrapped
    try:
        for _ in range(n_steps):
            actions.append(policy.select_action(dict(observation)))
    finally:
        del policy.predict_action_chunk
    return actions, chunk_calls


def action_shape_problems(policy: PreTrainedPolicy, observation: dict[str, torch.Tensor]) -> list[str]:
    """``select_action`` must return ``(batch, action_dim)`` finite values."""
    batch_size = next(iter(observation.values())).shape[0]
    action_dim = policy.config.action_feature.shape[0]
    action = seeded_action(policy, observation)
    problems = []
    if tuple(action.shape) != (batch_size, action_dim):
        problems.append(f"select_action shape {tuple(action.shape)}, expected {(batch_size, action_dim)}")
    if not torch.isfinite(action).all():
        problems.append("select_action returned non-finite values")
    return problems


# --- normalization --------------------------------------------------------------------


def normalizer_steps(pipeline) -> list:
    return [s for s in pipeline.steps if isinstance(s, NormalizerProcessorStep | UnnormalizerProcessorStep)]


def missing_normalization_stats(pipeline) -> list[str]:
    """Features a (un)normalizer step should scale but has no stats for.

    lerobot skips normalization silently for such keys (identity), which is how
    lerobot#4647 dropped eval reward from ~118-199 to ~4.8.
    """
    missing = []
    for step in normalizer_steps(pipeline):
        for key, ft in step.features.items():
            mode = step.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if mode is not NormalizationMode.IDENTITY and key not in step._tensor_stats:
                missing.append(key)
    return sorted(set(missing))


def processor_stats_problems(path: str | Path, config: PreTrainedConfig) -> list[str]:
    """Load saved pre/post processors from ``path`` and list features without stats."""
    pre, post = make_pre_post_processors(config, pretrained_path=str(path))
    return [f"preprocessor has no stats for {k}" for k in missing_normalization_stats(pre)] + [
        f"postprocessor has no stats for {k}" for k in missing_normalization_stats(post)
    ]
