import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  formatMetric,
  HEADLINE_METRICS,
  headlineMean,
  listRuns,
  loadRun,
  readReport,
} from '../src/core/runs';

const FIXTURE_DATA = path.join(__dirname, 'fixtures', 'data');

describe('listRuns', () => {
  it('finds runs at depth 1 and 2 under the data dir', () => {
    const names = listRuns(FIXTURE_DATA).map((r) => r.name);
    expect(names).toContain('run-a');
    expect(names).toContain('nested/run-b');
  });

  it('returns empty for a missing dir', () => {
    expect(listRuns(path.join(FIXTURE_DATA, 'nope'))).toEqual([]);
  });
});

describe('loadRun / headlineMean', () => {
  const run = loadRun(path.join(FIXTURE_DATA, 'run-a'));

  it('parses summary.json', () => {
    expect(run).not.toBeNull();
    expect(run!.summary.tasks_per_condition).toBe(2);
  });

  it('extracts headline means per condition', () => {
    expect(headlineMean(run!, 'heuristic', 'quality_score')).toBe(0.95);
    expect(headlineMean(run!, 'heuristic', 'missing_metric')).toBeNull();
    expect(headlineMean(run!, 'missing_condition', 'quality_score')).toBeNull();
  });

  it('includes queue depth and drains in the headline metrics', () => {
    expect(HEADLINE_METRICS).toContain('avg_queue_depth');
    expect(HEADLINE_METRICS).toContain('drain_count');
    expect(headlineMean(run!, 'heuristic', 'avg_queue_depth')).toBe(0.25);
    expect(headlineMean(run!, 'heuristic', 'drain_count')).toBe(0.5);
  });
});

describe('readReport', () => {
  it('reads report.txt when present', () => {
    expect(readReport(path.join(FIXTURE_DATA, 'run-a'))).toContain(
      'Baseline Performance Benchmark',
    );
  });

  it('returns null when absent', () => {
    expect(readReport(path.join(FIXTURE_DATA, 'nested', 'run-b'))).toBeNull();
  });
});

describe('formatMetric', () => {
  it('formats by metric kind', () => {
    expect(formatMetric('quality_score', 0.9)).toBe('0.900');
    expect(formatMetric('avg_queue_depth', 0.25)).toBe('0.250');
    expect(formatMetric('drain_count', 0.5)).toBe('0.5');
    expect(formatMetric('tool_cost_usd', 0.0042)).toBe('$0.0042');
    expect(formatMetric('total_tokens', 958.36)).toBe('958.4');
    expect(formatMetric('total_tokens', null)).toBe('-');
  });
});
