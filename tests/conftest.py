import os

# Must be set before huggingface_hub is imported anywhere: no test may touch the network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from policy_smoke.builders import OFFLINE_POLICIES, make_tiny_policy  # noqa: E402

torch.set_num_threads(min(4, os.cpu_count() or 1))


@pytest.fixture(params=OFFLINE_POLICIES)
def policy_name(request) -> str:
    return request.param


@pytest.fixture
def policy(policy_name):
    return make_tiny_policy(policy_name, seed=0)


@pytest.fixture
def checkpoint(policy, tmp_path) -> Path:
    """A tiny policy saved with lerobot's own ``save_pretrained``."""
    path = tmp_path / "ckpt"
    policy.save_pretrained(path)
    return path


def rewrite_weights(path: Path, drop: str | None = None, rename: tuple[str, str] | None = None) -> None:
    """Corrupt ``model.safetensors`` in place by dropping or renaming one key."""
    weights = path / "model.safetensors"
    tensors = load_file(weights)
    if drop is not None:
        del tensors[drop]
    if rename is not None:
        old, new = rename
        tensors[new] = tensors.pop(old)
    save_file(tensors, weights, metadata={"format": "pt"})
