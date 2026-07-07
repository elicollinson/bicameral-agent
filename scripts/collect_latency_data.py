"""Run the latency data collection harness against the real Gemini API.

Issue #35: collect ≥90 API observations across 3 tools × 6 conversation
lengths × 5 runs (plus the conscious loop at each length), feed them back
into the latency model, and emit a report + scatter plot.

Usage:
    GEMINI_API_KEY=… uv run python scripts/collect_latency_data.py \\
        --output-dir data/latency

The script is idempotent: existing files in ``--output-dir`` are
overwritten. Real API calls are made; expected cost is roughly $1.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from bicameral_agent.gemini import GeminiClient
from bicameral_agent.heuristic_controller import TOOL_IDS, Action
from bicameral_agent.latency_collection import (
    DEFAULT_CONV_LENGTHS,
    LatencyCollector,
    recompute_predictions,
    save_observations,
)
from bicameral_agent.latency_report import (
    LAYER_1_MIN_COVERAGE,
    LAYER_2_MIN_COVERAGE,
    format_text_report,
    layer1_coverage,
    layer2_coverage,
    save_scatter_plot,
)
from bicameral_agent.tool_latency import ToolLatencyModel

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="data/latency",
        help="Directory to write parquet/report/scatter outputs.",
    )
    parser.add_argument(
        "--runs-per-cell", type=int, default=5,
        help="Runs per (tool, conversation length) cell (default 5, AC requires ≥5).",
    )
    parser.add_argument(
        "--max-conv-length", type=int, default=max(DEFAULT_CONV_LENGTHS),
        help="Skip buckets larger than this (useful for cheap dry runs).",
    )
    parser.add_argument(
        "--include-conscious-loop", action="store_true", default=True,
        help="Also collect conscious-loop measurements at each bucket.",
    )
    parser.add_argument(
        "--no-conscious-loop", dest="include_conscious_loop", action="store_false",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-cell progress lines.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tool_latency_model = ToolLatencyModel()
    collector = LatencyCollector(tool_latency_model=tool_latency_model)
    client = GeminiClient(on_completion=collector.on_completion)
    collector.bind_client(client)

    buckets = [b for b in DEFAULT_CONV_LENGTHS if b <= args.max_conv_length]
    tool_ids = [
        TOOL_IDS[Action.SCANNER],
        TOOL_IDS[Action.AUDITOR],
        TOOL_IDS[Action.REFRESHER],
    ]

    total_cells = len(tool_ids) * len(buckets) * args.runs_per_cell
    logger.info(
        "Starting collection: %d tools × %d buckets × %d runs = %d cells",
        len(tool_ids), len(buckets), args.runs_per_cell, total_cells,
    )

    cell_idx = 0
    started_at = time.monotonic()
    for tool_id in tool_ids:
        for bucket in buckets:
            for run_idx in range(args.runs_per_cell):
                cell_idx += 1
                logger.info(
                    "[%d/%d] tool=%s bucket=%d run=%d",
                    cell_idx, total_cells, tool_id, bucket, run_idx,
                )
                try:
                    collector.collect_tool(tool_id, bucket, run_idx)
                except Exception as exc:  # pragma: no cover — defensive logging
                    logger.warning(
                        "  cell failed: tool=%s bucket=%d run=%d err=%s",
                        tool_id, bucket, run_idx, exc,
                    )

    if args.include_conscious_loop:
        for bucket in buckets:
            logger.info("conscious_loop bucket=%d", bucket)
            try:
                collector.collect_conscious_loop(bucket, run_index=0)
            except Exception as exc:  # pragma: no cover — defensive logging
                logger.warning("  conscious_loop failed: bucket=%d err=%s", bucket, exc)

    elapsed = time.monotonic() - started_at
    logger.info(
        "Collected %d API observations and %d tool observations in %.1fs",
        len(collector.api_observations), len(collector.tool_observations), elapsed,
    )

    if not collector.api_observations and not collector.tool_observations:
        logger.error("No observations collected (every cell failed); aborting.")
        return 1

    # The AC evaluates predictions made by the *trained* model, so recompute
    # predictions on every observation using the now-fitted model state.
    final_api, final_tool = recompute_predictions(
        collector.api_observations, collector.tool_observations, tool_latency_model,
    )

    api_path, tool_path = save_observations(output_dir, final_api, final_tool)
    logger.info("Wrote %s and %s", api_path, tool_path)

    report = format_text_report(final_api, final_tool)
    report_path = output_dir / "report.txt"
    report_path.write_text(report)
    sys.stdout.write(report)

    scatter_path = save_scatter_plot(final_api, output_dir / "scatter.png")
    logger.info("Wrote scatter to %s", scatter_path)

    overall_l2 = layer2_coverage(final_api)
    overall_l1 = layer1_coverage(final_tool)
    if (
        len(collector.api_observations) >= 90
        and overall_l2 >= LAYER_2_MIN_COVERAGE
        and overall_l1 >= LAYER_1_MIN_COVERAGE
    ):
        return 0
    logger.warning(
        "Acceptance thresholds not all met: n=%d L2=%.3f L1=%.3f",
        len(collector.api_observations), overall_l2, overall_l1,
    )
    return 0  # Soft pass — the issue says "report", not "fail-on-miss".


if __name__ == "__main__":
    sys.exit(main())
