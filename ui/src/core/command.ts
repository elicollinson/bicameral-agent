/** Assembly of the `uv run python scripts/run_baseline_benchmark.py ...` command.
 *
 * Data-driven: adding a new runner flag is one entry in FLAG_SPECS.
 * No experiment logic lives here — flags map 1:1 onto the Python CLI.
 */

export type Provider = 'gemini' | 'ollama';

export interface ExperimentConfig {
  provider: Provider;
  /** Model tag exactly as the user selected it (catalog name, never rewritten). */
  model: string;
  tasksPerCondition: number;
  maxTurns: number;
  /** Per-episode USD ceiling; null = runner default (no ceiling). */
  episodeBudget: number | null;
  /** Output dir for run artifacts, relative to the repo root. */
  outputDir: string;
  /** Optional hyperparameter TOML path (--config); null = bundled default. */
  configPath: string | null;
}

export const RUNNER_SCRIPT = 'scripts/run_baseline_benchmark.py';

/** The three benchmark conditions the runner always executes. */
export const CONDITIONS = ['no_subconscious', 'random', 'heuristic'] as const;

interface FlagSpec {
  flag: string;
  value: (c: ExperimentConfig) => string | number | null;
}

const FLAG_SPECS: FlagSpec[] = [
  { flag: '--output-dir', value: (c) => c.outputDir },
  { flag: '--provider', value: (c) => c.provider },
  { flag: '--model', value: (c) => c.model || null },
  { flag: '--tasks-per-condition', value: (c) => c.tasksPerCondition },
  { flag: '--max-turns', value: (c) => c.maxTurns },
  { flag: '--episode-budget', value: (c) => c.episodeBudget },
  { flag: '--config', value: (c) => c.configPath },
];

/** Arguments passed to `uv` (i.e. everything after the `uv` executable). */
export function buildRunnerArgs(config: ExperimentConfig): string[] {
  const args = ['run', 'python', RUNNER_SCRIPT];
  for (const spec of FLAG_SPECS) {
    const value = spec.value(config);
    if (value !== null && value !== '') {
      args.push(spec.flag, String(value));
    }
  }
  return args;
}

/** Human-readable command line for display before launch. */
export function formatCommand(args: string[]): string {
  return ['uv', ...args].join(' ');
}

export function defaultOutputDir(now: Date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  const stamp =
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  return `data/run-${stamp}`;
}
