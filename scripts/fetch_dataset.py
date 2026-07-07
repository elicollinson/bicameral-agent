#!/usr/bin/env python3
"""Fetch an external evaluation dataset into a local (git-ignored) cache.

Generalizes ``fetch_hard_benchmark.py`` to every dataset registered in
``bicameral_agent.eval_datasets``. The raw data is NOT redistributed in this
repo; this script pulls a subset from upstream into ``data/external/<name>.json``
for local use. Stdlib-only (urllib) -- no extra dependency.

Usage:
    python scripts/fetch_dataset.py --dataset frames [--limit N] [--out PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bicameral_agent.eval_datasets import build_dataset, dataset_names

# The builtin pool ships inside the package; there is nothing to fetch.
_FETCHABLE = [name for name in dataset_names() if name != "builtin"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hard_benchmark", choices=_FETCHABLE,
                        help="Dataset to fetch (default: hard_benchmark).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Tasks to pull (default: the dataset's default_limit).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Cache path override (default: data/external/<name>.json).")
    args = parser.parse_args(argv)

    dataset = build_dataset(args.dataset, cache_path=args.out)
    meta = dataset.meta
    print(f"{meta.name}: source={meta.source}")
    print(f"  license: {meta.license}")
    if meta.citation:
        print(f"  cite: {meta.citation}")
    if meta.requires_hf_token and not os.environ.get("HF_TOKEN"):
        print(
            f"  note: {meta.name} is gated on Hugging Face; accept its terms "
            "and set HF_TOKEN before fetching.",
            file=sys.stderr,
        )

    tasks = dataset.build(args.limit)
    dist = Counter(t.difficulty.value for t in tasks)
    print(f"Wrote {len(tasks)} tasks ({dict(dist)}) to {dataset.cache_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
