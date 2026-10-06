"""``policy-smoke check <dir>``: run offline invariants on a local LeRobot checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

# Set before lerobot / huggingface_hub are imported (they are imported lazily below).
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _run(name: str, fn: Callable[[], list[str]]) -> tuple[str, str]:
    try:
        problems = fn()
    except Exception as e:  # report, don't crash: one broken check should not hide the others
        return "FAIL", f"{name}: {type(e).__name__}: {e}"
    if problems:
        more = " ..." if len(problems) > 5 else ""
        return "FAIL", f"{name}: " + "; ".join(problems[:5]) + more
    return "PASS", name


def check_checkpoint(path: Path) -> list[tuple[str, str]]:
    """Run all checkpoint checks; return ``(status, message)`` lines."""
    from lerobot.policies.factory import get_policy_class

    from policy_smoke import checks
    from policy_smoke.builders import make_observation

    results = [_run("files present and non-empty", lambda: checks.checkpoint_file_problems(path))]
    if results[-1][0] == "FAIL":
        return results

    config = checks.load_config_offline(path)

    def key_mismatches() -> list[str]:
        report = checks.load_strict_report(get_policy_class(config.type), path, config)
        return [f"missing key {k}" for k in report.missing] + [
            f"unexpected key {k}" for k in report.unexpected
        ]

    results.append(_run("weights match model (no missing/unexpected keys)", key_mismatches))
    if results[-1][0] == "FAIL":
        return results

    policy = checks.load_policy_strict(path)
    obs = make_observation(policy.config, batch_size=2)
    results.append(_run("save/reload keeps weights and config", lambda: checks.roundtrip_differences(policy)))
    results.append(_run("select_action shape", lambda: checks.action_shape_problems(policy, obs)))

    if (path / checks.PREPROCESSOR_FILE).is_file():
        results.append(_run("normalization stats", lambda: checks.processor_stats_problems(path, config)))
    else:
        results.append(("SKIP", f"normalization stats: no {checks.PREPROCESSOR_FILE}"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="policy-smoke", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check a local checkpoint directory")
    check.add_argument("path", type=Path, help="directory written by policy.save_pretrained()")
    args = parser.parse_args(argv)

    if not args.path.is_dir():
        print(f"FAIL {args.path} is not a directory")
        return 1
    results = check_checkpoint(args.path)
    for status, message in results:
        print(f"{status} {message}")
    failed = sum(status == "FAIL" for status, _ in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
