import React, { useMemo, useState } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import Configure from './screens/Configure';
import Review from './screens/Review';
import Running from './screens/Running';
import type { ExperimentConfig } from './core/command';
import { loadPricingKeys } from './core/pricing';
import { findRepoRoot } from './core/repo';
import { startRun, type RunSession } from './core/session';

type Screen = 'menu' | 'configure' | 'running' | 'review';

interface Props {
  repoRoot?: string;
}

export default function App({ repoRoot: repoRootProp }: Props) {
  const { exit } = useApp();
  const repoRoot = useMemo(
    () => repoRootProp ?? findRepoRoot(),
    [repoRootProp],
  );
  const pricingKeys = useMemo(() => loadPricingKeys(repoRoot), [repoRoot]);

  const [screen, setScreen] = useState<Screen>('menu');
  const [session, setSession] = useState<RunSession | null>(null);

  const runActive = session !== null && session.running;

  const items = [
    {
      key: 'new',
      label: runActive
        ? 'New experiment (blocked: a run is active)'
        : 'New experiment',
      disabled: runActive,
    },
    ...(session !== null
      ? [
          {
            key: 'attach',
            label: runActive ? 'Attach to running experiment' : 'Last run',
            disabled: false,
          },
        ]
      : []),
    { key: 'review', label: 'Review runs', disabled: false },
    { key: 'quit', label: 'Quit', disabled: false },
  ];
  const [menuIndex, setMenuIndex] = useState(0);
  const clamped = Math.min(menuIndex, items.length - 1);

  useInput(
    (input, key) => {
      if (key.upArrow) setMenuIndex(Math.max(clamped - 1, 0));
      else if (key.downArrow) {
        setMenuIndex(Math.min(clamped + 1, items.length - 1));
      } else if (input === 'q') exit();
      else if (key.return) {
        const item = items[clamped];
        if (item.disabled) return;
        if (item.key === 'new') setScreen('configure');
        else if (item.key === 'attach') setScreen('running');
        else if (item.key === 'review') setScreen('review');
        else if (item.key === 'quit') {
          session?.kill();
          exit();
        }
      }
    },
    { isActive: screen === 'menu' },
  );

  const launch = (config: ExperimentConfig) => {
    setSession(startRun(config, repoRoot));
    setScreen('running');
  };

  if (screen === 'configure') {
    return (
      <Configure
        pricingKeys={pricingKeys}
        onLaunch={launch}
        onCancel={() => setScreen('menu')}
      />
    );
  }
  if (screen === 'running' && session !== null) {
    return (
      <Running
        session={session}
        onBack={() => setScreen('menu')}
        onReview={() => setScreen('review')}
      />
    );
  }
  if (screen === 'review') {
    return <Review repoRoot={repoRoot} onBack={() => setScreen('menu')} />;
  }

  return (
    <Box flexDirection="column">
      <Text bold color="cyan">
        bicameral-agent experiment console
      </Text>
      <Text dimColor>repo: {repoRoot}</Text>
      {items.map((item, i) => (
        <Text
          key={item.key}
          color={i === clamped ? 'green' : undefined}
          dimColor={item.disabled}
        >
          {(i === clamped ? '❯ ' : '  ') + item.label}
        </Text>
      ))}
      <Text dimColor>enter select · ↑/↓ move · q quit</Text>
    </Box>
  );
}
