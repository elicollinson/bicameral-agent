import { describe, expect, it } from 'vitest';
import {
  buildRunnerArgs,
  defaultOutputDir,
  formatCommand,
  type ExperimentConfig,
} from '../src/core/command';

const base: ExperimentConfig = {
  provider: 'ollama',
  model: 'gemma4:31b',
  tasksPerCondition: 2,
  maxTurns: 10,
  episodeBudget: 0.5,
  outputDir: 'data/run-x',
  configPath: null,
};

describe('buildRunnerArgs', () => {
  it('assembles the full uv command against the current CLI flags', () => {
    expect(buildRunnerArgs(base)).toEqual([
      'run',
      'python',
      'scripts/run_baseline_benchmark.py',
      '--output-dir',
      'data/run-x',
      '--provider',
      'ollama',
      '--model',
      'gemma4:31b',
      '--tasks-per-condition',
      '2',
      '--max-turns',
      '10',
      '--episode-budget',
      '0.5',
    ]);
  });

  it('passes the model tag verbatim (no -cloud rewriting)', () => {
    const args = buildRunnerArgs({ ...base, model: 'glm-5.2' });
    expect(args).toContain('glm-5.2');
    expect(args.join(' ')).not.toContain('-cloud');
  });

  it('omits optional flags when unset', () => {
    const args = buildRunnerArgs({
      ...base,
      model: '',
      episodeBudget: null,
      configPath: null,
    });
    expect(args).not.toContain('--model');
    expect(args).not.toContain('--episode-budget');
    expect(args).not.toContain('--config');
  });

  it('includes --config when a TOML path is given', () => {
    const args = buildRunnerArgs({ ...base, configPath: 'hyper.toml' });
    expect(args).toContain('--config');
    expect(args).toContain('hyper.toml');
  });
});

describe('formatCommand', () => {
  it('prefixes uv', () => {
    expect(formatCommand(['run', 'python'])).toBe('uv run python');
  });
});

describe('defaultOutputDir', () => {
  it('stamps a data/run-* dir', () => {
    expect(defaultOutputDir(new Date(2026, 6, 6, 9, 5, 3))).toBe(
      'data/run-20260706-090503',
    );
  });
});
