import React from 'react';
import { render } from 'ink-testing-library';
import { describe, expect, it } from 'vitest';
import Configure from '../src/screens/Configure';
import type { ExperimentConfig } from '../src/core/command';

// With no pricing keys the gemini model step is a free-text field, so the
// wizard can be driven end-to-end without any catalog fetch.
function renderWizard(onLaunch: (c: ExperimentConfig) => void = () => {}) {
  return render(
    <Configure pricingKeys={[]} onLaunch={onLaunch} onCancel={() => {}} />,
  );
}

const tick = () => new Promise((r) => setTimeout(r, 20));

/** Drive the wizard up to (but not past) the parallel-episodes step. */
async function stepToParallel(stdin: { write: (s: string) => void }) {
  stdin.write('\r'); // provider: gemini
  await tick();
  stdin.write('m'); // model tag (free text)
  await tick();
  stdin.write('\r');
  await tick();
  stdin.write('\r'); // tasks: default 5
  await tick();
  stdin.write('\r'); // max turns: default 10
  await tick();
}

describe('Configure wizard: parallel episodes', () => {
  it('shows the step with default 1 and a provider-concurrency hint', async () => {
    const { lastFrame, stdin, unmount } = renderWizard();
    await stepToParallel(stdin);
    const frame = lastFrame() ?? '';
    expect(frame).toContain('Parallel episodes');
    expect(frame).toContain('1');
    expect(frame).toContain("provider's concurrent-request allowance");
    unmount();
  });

  it('threads the value into the confirm summary and launch config', async () => {
    const launches: ExperimentConfig[] = [];
    const { lastFrame, stdin, unmount } = renderWizard((c) => {
      launches.push(c);
    });
    await stepToParallel(stdin);
    stdin.write('0'); // append to default "1" → 10
    await tick();
    stdin.write('\r');
    await tick();
    stdin.write('\r'); // budget: none
    await tick();
    stdin.write('\r'); // output dir: default
    await tick();
    const frame = lastFrame() ?? '';
    expect(frame).toContain('parallel=10');
    expect(frame).toContain('--parallel-episodes 10');
    stdin.write('\r'); // launch
    await tick();
    expect(launches).toHaveLength(1);
    expect(launches[0].parallelEpisodes).toBe(10);
    unmount();
  });

  it('rejects zero', async () => {
    const { lastFrame, stdin, unmount } = renderWizard();
    await stepToParallel(stdin);
    stdin.write('\x7f'); // backspace the default "1"
    await tick();
    stdin.write('0');
    await tick();
    stdin.write('\r');
    await tick();
    const frame = lastFrame() ?? '';
    expect(frame).toContain('Parallel episodes');
    expect(frame).toContain('enter a positive whole number');
    unmount();
  });
});
