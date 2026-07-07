#!/usr/bin/env python3
"""Fetch the harder external benchmark subset into a local (git-ignored) cache.

The raw data is NOT redistributed in this repo (see docs/hard_benchmark.md for
sources, licenses, and attribution). This script pulls subsets from upstream
into ``data/external/hard_benchmark.json`` for local use; that path is
git-ignored. Stdlib-only (urllib) -- no extra dependency.

Usage:
    python scripts/fetch_hard_benchmark.py [--frames N] [--crepe N] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bicameral_agent.hard_benchmark import _DEFAULT_CACHE, build_hard_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100, help="FRAMES (hard) tasks to pull.")
    parser.add_argument("--crepe", type=int, default=60, help="CREPE (tricky) tasks to pull.")
    parser.add_argument("--out", type=Path, default=_DEFAULT_CACHE)
    args = parser.parse_args(argv)

    print(f"Fetching {args.frames} FRAMES + {args.crepe} CREPE tasks -> {args.out}")
    tasks = build_hard_benchmark(args.frames, args.crepe, args.out)
    dist = Counter(t.difficulty.value for t in tasks)
    print(f"Wrote {len(tasks)} tasks ({dict(dist)}) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
