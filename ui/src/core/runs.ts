/** Read-only access to completed run artifacts under data/.
 *
 * A "run" is any directory (scanned up to two levels below the data dir)
 * containing a summary.json written by scripts/run_baseline_benchmark.py.
 * The UI never mutates run artifacts.
 */

import fs from 'node:fs';
import path from 'node:path';

export interface MetricSummary {
  mean: number;
  std: number;
  ci_lower: number;
  ci_upper: number;
  n: number;
}

export interface ConditionReport {
  condition: string;
  n_episodes: number;
  summaries: Record<string, MetricSummary>;
}

export interface RunSummary {
  tasks_per_condition?: number;
  max_turns?: number;
  conditions: Record<string, ConditionReport>;
}

export interface RunInfo {
  /** Path relative to the data dir, e.g. "baseline". */
  name: string;
  dir: string;
  mtimeMs: number;
  summary: RunSummary;
}

/** Metrics shown in run detail tables and the two-run comparison. */
export const HEADLINE_METRICS = [
  'quality_score',
  'total_tokens',
  'total_turns',
  'wall_clock_ms',
  'tool_cost_usd',
  'avg_queue_depth',
  'drain_count',
] as const;

export function loadRun(dir: string, name?: string): RunInfo | null {
  const summaryPath = path.join(dir, 'summary.json');
  try {
    const stat = fs.statSync(summaryPath);
    const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8')) as RunSummary;
    if (typeof summary !== 'object' || summary === null || !summary.conditions) {
      return null;
    }
    return { name: name ?? path.basename(dir), dir, mtimeMs: stat.mtimeMs, summary };
  } catch {
    return null;
  }
}

/** List completed runs under `dataDir`, most recent first. */
export function listRuns(dataDir: string): RunInfo[] {
  const runs: RunInfo[] = [];
  const visit = (dir: string, name: string, depth: number) => {
    const run = loadRun(dir, name);
    if (run) {
      runs.push(run);
      return;
    }
    if (depth >= 2) return;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const childName = name ? `${name}/${entry.name}` : entry.name;
      visit(path.join(dir, entry.name), childName, depth + 1);
    }
  };
  visit(dataDir, '', 0);
  return runs.sort((a, b) => b.mtimeMs - a.mtimeMs);
}

export function readReport(dir: string): string | null {
  try {
    return fs.readFileSync(path.join(dir, 'report.txt'), 'utf8');
  } catch {
    return null;
  }
}

/** Mean of a headline metric for one condition, or null if absent. */
export function headlineMean(
  run: RunInfo,
  condition: string,
  metric: string,
): number | null {
  const mean = run.summary.conditions[condition]?.summaries?.[metric]?.mean;
  return typeof mean === 'number' ? mean : null;
}

export function formatMetric(metric: string, value: number | null): string {
  if (value === null) return '-';
  if (metric === 'quality_score' || metric === 'avg_queue_depth') {
    return value.toFixed(3);
  }
  if (metric === 'tool_cost_usd') return `$${value.toFixed(4)}`;
  return value.toFixed(1);
}
