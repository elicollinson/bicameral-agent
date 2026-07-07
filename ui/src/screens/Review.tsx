import React, { useMemo, useState } from 'react';
import path from 'node:path';
import { Box, Text, useInput } from 'ink';
import {
  formatMetric,
  HEADLINE_METRICS,
  headlineMean,
  listRuns,
  readReport,
  type RunInfo,
} from '../core/runs';

const REPORT_PAGE = 14;

type Mode = 'list' | 'detail' | 'compare';

interface Props {
  repoRoot: string;
  onBack: () => void;
}

function MetricsTable({ run }: { run: RunInfo }) {
  const conditions = Object.keys(run.summary.conditions);
  return (
    <Box flexDirection="column">
      <Text bold>
        {'metric'.padEnd(16)}
        {conditions.map((c) => c.padStart(18)).join('')}
      </Text>
      {HEADLINE_METRICS.map((metric) => (
        <Text key={metric}>
          {metric.padEnd(16)}
          {conditions
            .map((c) => formatMetric(metric, headlineMean(run, c, metric)).padStart(18))
            .join('')}
        </Text>
      ))}
    </Box>
  );
}

function CompareTable({ a, b }: { a: RunInfo; b: RunInfo }) {
  const conditions = [
    ...new Set([
      ...Object.keys(a.summary.conditions),
      ...Object.keys(b.summary.conditions),
    ]),
  ];
  return (
    <Box flexDirection="column">
      <Text bold>
        {'condition / metric'.padEnd(34)}
        {a.name.slice(-16).padStart(18)}
        {b.name.slice(-16).padStart(18)}
      </Text>
      {conditions.map((condition) => (
        <Box key={condition} flexDirection="column">
          <Text bold color="cyan">
            {condition}
          </Text>
          {HEADLINE_METRICS.map((metric) => (
            <Text key={metric}>
              {`  ${metric}`.padEnd(34)}
              {formatMetric(metric, headlineMean(a, condition, metric)).padStart(18)}
              {formatMetric(metric, headlineMean(b, condition, metric)).padStart(18)}
            </Text>
          ))}
        </Box>
      ))}
    </Box>
  );
}

export default function Review({ repoRoot, onBack }: Props) {
  const runs = useMemo(() => listRuns(path.join(repoRoot, 'data')), [repoRoot]);
  const [mode, setMode] = useState<Mode>('list');
  const [index, setIndex] = useState(0);
  const [marked, setMarked] = useState<string[]>([]);
  const [scroll, setScroll] = useState(0);

  const current: RunInfo | undefined = runs[index];
  const markedRuns = runs.filter((r) => marked.includes(r.dir));
  const report =
    mode === 'detail' && current ? readReport(current.dir) : null;
  const reportLines = report ? report.split('\n') : [];

  useInput((input, key) => {
    if (mode === 'list') {
      if (key.escape) onBack();
      else if (key.upArrow) setIndex((i) => Math.max(i - 1, 0));
      else if (key.downArrow) setIndex((i) => Math.min(i + 1, runs.length - 1));
      else if (key.return && current) {
        setScroll(0);
        setMode('detail');
      } else if (input === ' ' && current) {
        setMarked((m) =>
          m.includes(current.dir)
            ? m.filter((d) => d !== current.dir)
            : m.length < 2
              ? [...m, current.dir]
              : m,
        );
      } else if (input === 'c' && markedRuns.length === 2) {
        setMode('compare');
      }
    } else if (mode === 'detail') {
      if (key.escape) setMode('list');
      else if (key.upArrow) setScroll((s) => Math.max(s - 1, 0));
      else if (key.downArrow) {
        setScroll((s) =>
          Math.min(s + 1, Math.max(reportLines.length - REPORT_PAGE, 0)),
        );
      }
    } else if (key.escape) {
      setMode('list');
    }
  });

  if (mode === 'compare' && markedRuns.length === 2) {
    return (
      <Box flexDirection="column">
        <Text bold color="cyan">
          Compare: {markedRuns[0].name} vs {markedRuns[1].name}
        </Text>
        <CompareTable a={markedRuns[0]} b={markedRuns[1]} />
        <Text dimColor>esc back</Text>
      </Box>
    );
  }

  if (mode === 'detail' && current) {
    return (
      <Box flexDirection="column">
        <Text bold color="cyan">
          Run: {current.name}
        </Text>
        <Text dimColor>
          {current.dir} · tasks/condition=
          {current.summary.tasks_per_condition ?? '?'} · max-turns=
          {current.summary.max_turns ?? '?'}
        </Text>
        <MetricsTable run={current} />
        <Box flexDirection="column" marginTop={1}>
          <Text dimColor>── report.txt ──</Text>
          {report === null ? (
            <Text dimColor>(no report.txt in this run)</Text>
          ) : (
            reportLines
              .slice(scroll, scroll + REPORT_PAGE)
              .map((line, i) => (
                <Text key={scroll + i} wrap="truncate-end">
                  {line}
                </Text>
              ))
          )}
        </Box>
        <Text dimColor>↑/↓ scroll report · esc back</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column">
      <Text bold color="cyan">
        Review runs (data/)
      </Text>
      {runs.length === 0 && (
        <Text dimColor>No completed runs found (no summary.json under data/).</Text>
      )}
      {runs.map((run, i) => {
        const mark = marked.includes(run.dir) ? '[x]' : '[ ]';
        return (
          <Text key={run.dir} color={i === index ? 'green' : undefined}>
            {(i === index ? '❯ ' : '  ') + mark} {run.name.padEnd(32)}
            <Text dimColor>{new Date(run.mtimeMs).toLocaleString()}</Text>
          </Text>
        );
      })}
      <Text dimColor>
        enter open · space mark ({marked.length}/2) · c compare marked · esc
        menu
      </Text>
    </Box>
  );
}
