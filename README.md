# policy-smoke

CPU-only smoke tests for [LeRobot](https://github.com/huggingface/lerobot) policies that catch "loads fine, silently degraded" bugs. No GPU, no model downloads, about 20 seconds on a laptop.

Every test builds a tiny, randomly initialised policy from a config object (6-dim state, 6-dim action, one 3x64x64 image or none), saves it with `save_pretrained`, reloads it with `from_pretrained`, and checks invariants. `HF_HUB_OFFLINE=1` is set for the whole run.

## What it catches

| Check | Test file | Failure it guards against |
|---|---|---|
| 1. Round trip: same `state_dict` (keys and values) and same config fields after save/load | `test_roundtrip.py` | Weights or config fields lost on save/load |
| 2. Determinism: same seed and input give the same action before and after round trip | `test_roundtrip.py` | Hidden state or buffers not saved |
| 3. Missing or renamed weight keys are reported by `load_strict_report()` | `test_strict_load.py` | [#4711](https://github.com/huggingface/lerobot/issues/4711): `load_state_dict(strict=False)` silently drops keys |
| 4. `PreTrainedConfig.from_pretrained(path)` sets `pretrained_path` | `test_pretrained_path.py` | [#4647](https://github.com/huggingface/lerobot/issues/4647): processors built with empty stats, reward ~118-199 drops to ~4.8 |
| 5. Normalization stats survive processor save/load; processors without stats are flagged | `test_normalization.py` | [#4647](https://github.com/huggingface/lerobot/issues/4647): normalization quietly becomes identity |
| 6. `select_action` returns `(batch, action_dim)`; a new chunk is predicted every `n_action_steps` calls | `test_shapes.py` | Wrong chunk slicing or axis order |
| Config knob has an effect: `use_tanh_squash=False` gives unbounded actions | `test_config_knobs.py` | [#4727](https://github.com/huggingface/lerobot/issues/4727): `use_tanh_squash` accepted but never read |
| CLI: `config.json` and `model.safetensors` exist and are not empty | `test_cli.py` | [#4649](https://github.com/huggingface/lerobot/issues/4649): `migrate_policy_normalization` writes an empty `config.json` and no weights |

Known upstream bugs are marked `xfail(strict=True, raises=AssertionError)`. When upstream fixes one, that test passes, the suite turns red, and the marker should be removed.

## Run the tests

```bash
python3.12 -m venv .venv && source .venv/bin/activate
# optional, smaller download: CPU-only torch
pip install "torch<2.12" "torchvision<0.27" --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
```

Sample run (lerobot 0.6.1, torch 2.11.0, 4 CPU cores):

```
XFAIL tests/test_config_knobs.py::test_tanh_squash_off_leaves_actions_unbounded - lerobot#4727: GaussianActor ignores use_tanh_squash=False
XFAIL tests/test_pretrained_path.py::test_config_from_pretrained_sets_pretrained_path[act] - lerobot#4647: PreTrainedConfig.from_pretrained leaves pretrained_path=None
...
XFAIL tests/test_pretrained_path.py::test_processors_from_loaded_config_keep_stats[vqbet] - lerobot#4647: PreTrainedConfig.from_pretrained leaves pretrained_path=None
XFAIL tests/test_shapes.py::test_chunk_is_batch_major[tdmpc] - tdmpc predict_action_chunk returns (horizon, batch, action_dim), not (batch, ...)
62 passed, 2 skipped, 12 xfailed in 16.97s
```

## Check your own checkpoint

```bash
policy-smoke check path/to/pretrained_model
```

It runs offline on CPU and checks: files present and non-empty; every weight key matches the model (no missing or unexpected keys); save/reload keeps weights and config; `select_action` output shape; and, if `policy_preprocessor.json` is present, that every normalized feature has stats. It exits 1 if any check fails.

```
$ policy-smoke check ./good
PASS files present and non-empty
PASS weights match model (no missing/unexpected keys)
PASS save/reload keeps weights and config
PASS select_action shape
PASS normalization stats

$ policy-smoke check ./no_stats
PASS files present and non-empty
PASS weights match model (no missing/unexpected keys)
PASS save/reload keeps weights and config
PASS select_action shape
FAIL normalization stats: preprocessor has no stats for action; preprocessor has no stats for observation.image; preprocessor has no stats for observation.state; postprocessor has no stats for action

$ policy-smoke check ./dropped_key
PASS files present and non-empty
FAIL weights match model (no missing/unexpected keys): missing key model.action_head.weight
```

The helpers are also importable: `policy_smoke.checks.load_strict_report(policy_cls, path)`, `missing_normalization_stats(pipeline)`, `state_dict_differences(a, b)`, and the tiny-config factories in `policy_smoke.builders`.

## Limitations

- Tested against **lerobot 0.6.1** with torch 2.11.0 on Python 3.12. CI runs both 0.6.1 and the latest release.
- Policies covered: `act`, `diffusion`, `vqbet`, `tdmpc`, `gaussian_actor` (ACT, Diffusion and VQ-BeT with `pretrained_backbone_weights=None`).
- Not covered, because they cannot be built offline on CPU (tried with `HF_HUB_OFFLINE=1`):
  - `smolvla`, `multi_task_dit`, `eo1`, `evo1`, `vla_jepa`: the constructor downloads a VLM or text encoder from huggingface.co.
  - `groot`, `molmoact2`: the constructor needs a Hub snapshot.
  - `pi0`, `pi05`, `lingbot_va`: build a multi-billion-parameter model; did not finish in 120 s on CPU.
  - `pi0_fast`: needs `scipy` and loads its tokenizers from huggingface.co.
  - `wall_x`: needs `peft` and builds on Qwen2.5-VL.
  - `xvla`: needs a full Florence-2 `vision_config`.
  - `fastwam`: builds on the Wan2.2-TI2V-5B video model (`action_dim` defaults to 7; not tried with a smaller config).
- Weights are random, so these checks prove plumbing (save, load, shapes, stats), not task performance.
- `policy-smoke check` turns off `pretrained_backbone_weights` before building the model, so it never fetches torchvision weights. The checkpoint's own `model.safetensors` supplies those weights, and the key check confirms nothing is missing.
- Language-conditioned policies need a `task` string in the observation; the CLI does not provide one.

## License

MIT
