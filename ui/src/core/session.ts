/** A running benchmark child process plus everything observed about it.
 *
 * The session outlives the Running screen: the App owns it, so the user can
 * go back to the menu and re-attach without killing the run.
 */

import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import path from 'node:path';
import {
  buildRunnerArgs,
  CONDITIONS,
  type ExperimentConfig,
} from './command';
import { parquetRowCount } from './parquet';
import { applyLine, initialProgress, type ProgressState } from './progress';

const MAX_TAIL_LINES = 200;
const POLL_INTERVAL_MS = 1500;

export interface RunSession {
  config: ExperimentConfig;
  args: string[];
  startedAt: number;
  lines: string[];
  progress: ProgressState;
  episodeCounts: Record<string, number>;
  exitCode: number | null;
  spawnError: string | null;
  subscribe(listener: () => void): () => void;
  kill(): void;
  readonly running: boolean;
}

export function startRun(
  config: ExperimentConfig,
  repoRoot: string,
): RunSession {
  const args = buildRunnerArgs(config);
  const listeners = new Set<() => void>();
  const notify = () => listeners.forEach((fn) => fn());

  let child: ChildProcessWithoutNullStreams | null = null;
  let pollTimer: ReturnType<typeof setInterval> | null = null;

  const session: RunSession = {
    config,
    args,
    startedAt: Date.now(),
    lines: [],
    progress: initialProgress(),
    episodeCounts: {},
    exitCode: null,
    spawnError: null,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    kill() {
      child?.kill('SIGTERM');
    },
    get running() {
      return this.exitCode === null && this.spawnError === null;
    },
  };

  const pushChunk = (chunk: Buffer) => {
    for (const line of chunk.toString('utf8').split('\n')) {
      if (line.trim() === '') continue;
      session.lines.push(line);
      session.progress = applyLine(session.progress, line);
    }
    if (session.lines.length > MAX_TAIL_LINES) {
      session.lines.splice(0, session.lines.length - MAX_TAIL_LINES);
    }
    notify();
  };

  const stopPolling = () => {
    if (pollTimer !== null) clearInterval(pollTimer);
    pollTimer = null;
  };

  const pollArtifacts = async () => {
    const outputDir = path.resolve(repoRoot, config.outputDir);
    let changed = false;
    for (const condition of CONDITIONS) {
      const count = await parquetRowCount(
        path.join(outputDir, `${condition}.parquet`),
      );
      if (count !== null && count !== session.episodeCounts[condition]) {
        session.episodeCounts[condition] = count;
        changed = true;
      }
    }
    if (changed) notify();
  };

  child = spawn('uv', args, {
    cwd: repoRoot,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  });
  child.stdout.on('data', pushChunk);
  child.stderr.on('data', pushChunk);
  child.on('error', (err) => {
    session.spawnError = err.message;
    stopPolling();
    notify();
  });
  child.on('close', (code, signal) => {
    session.exitCode = code ?? (signal ? 1 : 0);
    stopPolling();
    // One last poll so the final episode counts land.
    void pollArtifacts();
  });

  pollTimer = setInterval(() => void pollArtifacts(), POLL_INTERVAL_MS);
  pollTimer.unref?.();

  return session;
}
