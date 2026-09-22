#!/usr/bin/env python3
"""Run held-out native-asset planning or evaluation for all three study arms."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "evaluate"))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    if not parsed.args:
        parser.error("pass the evaluator arguments after plan/evaluate")
    for arm in ("native", "native_plus_ours", "ours"):
        print(f"=== {parsed.command}: arm={arm} ===", flush=True)
        subprocess.run(
            [sys.executable, "-m", "openpi.robocasa365.evaluate", parsed.command, "--arm", arm, *parsed.args],
            check=True,
        )


if __name__ == "__main__":
    main()
