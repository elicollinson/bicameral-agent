import React, { useEffect, useState } from 'react';
import { Box, Text, useInput } from 'ink';
import { CONDITIONS } from '../core/command';
import type { RunSession } from '../core/session';

const TAIL_LINES = 12;
const BAR_WIDTH = 24;

function bar(count: number, total: number): string {
  if (total <= 0) return '';
  const filled = Math.round(Math.min(count / total, 1) * BAR_WIDTH);
  return '█'.repeat(filled) + '░'.repeat(BAR_WIDTH - filled);
}

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  const mm = String(Math.floor(s / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

interface Props {
  session: RunSession;
  onBack: () => void;
  onReview: () => void;
}

export default function Running({ session, onBack, onReview }: Props) {
  const [, setTick] = useState(0);

  // Re-render on session updates and once a second for the elapsed clock.
  useEffect(() => {
    const unsubscribe = session.subscribe(() => setTick((t) => t + 1));
    const clock = setInterval(() => setTick((t) => t + 1), 1000);
    return () => {
      unsubscribe();
      clearInterval(clock);
    };
  }, [session]);

  useInput((input, key) => {
    if (key.escape) onBack();
    else if (input === 'k' && session.running) session.kill();
    else if (input === 'r' && !session.running) onReview();
  });

  const { config, progress, episodeCounts } = session;
  const total = progress.totalPerCondition ?? config.tasksPerCondition;
  const done = !session.running;

  return (
    <Box flexDirection="column">
      <Text bold color="cyan">
        Running: {config.provider}/{config.model} → {config.outputDir}
      </Text>
      <Text dimColor>
        elapsed {formatElapsed(Date.now() - session.startedAt)}
        {session.spawnError && ` · spawn failed: ${session.spawnError}`}
        {session.exitCode !== null && ` · exited with code ${session.exitCode}`}
      </Text>
      <Box flexDirection="column" marginTop={1}>
        {CONDITIONS.map((condition) => {
          const count = episodeCounts[condition] ?? 0;
          const active = progress.activeCondition === condition && !done;
          const cost = progress.costByCondition[condition];
          return (
            <Text key={condition} color={active ? 'green' : undefined}>
              {(active ? '▶ ' : '  ') + condition.padEnd(17)}
              {bar(count, total)} {count}/{total}
              {cost !== undefined && (
                <Text dimColor>{`  mean cost $${cost.toFixed(4)}/ep`}</Text>
              )}
            </Text>
          );
        })}
      </Box>
      <Box flexDirection="column" marginTop={1}>
        <Text dimColor>── runner output ──</Text>
        {session.lines.slice(-TAIL_LINES).map((line, i) => (
          <Text key={i} dimColor wrap="truncate-end">
            {line}
          </Text>
        ))}
      </Box>
      <Box marginTop={1}>
        <Text dimColor>
          {done
            ? 'r review results · esc menu'
            : 'k kill run · esc menu (run keeps going)'}
        </Text>
      </Box>
    </Box>
  );
}
