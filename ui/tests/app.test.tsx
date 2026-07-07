import path from 'node:path';
import React from 'react';
import { render } from 'ink-testing-library';
import { describe, expect, it } from 'vitest';
import App from '../src/app';

// The app must start and render with no Python process present; pointing the
// repo root at an empty fixture tree exercises the degraded paths too.
const EMPTY_ROOT = path.join(__dirname, 'fixtures');

describe('App', () => {
  it('renders the main menu without a Python process', () => {
    const { lastFrame, unmount } = render(<App repoRoot={EMPTY_ROOT} />);
    const frame = lastFrame() ?? '';
    expect(frame).toContain('bicameral-agent experiment console');
    expect(frame).toContain('New experiment');
    expect(frame).toContain('Review runs');
    unmount();
  });

  it('opens the review screen and lists fixture runs', async () => {
    const { lastFrame, stdin, unmount } = render(<App repoRoot={EMPTY_ROOT} />);
    stdin.write('[B'); // down → Review runs
    await new Promise((r) => setTimeout(r, 20));
    stdin.write('\r');
    await new Promise((r) => setTimeout(r, 20));
    const frame = lastFrame() ?? '';
    expect(frame).toContain('Review runs');
    expect(frame).toContain('run-a');
    unmount();
  });

  it('opens the configure wizard at the provider step', async () => {
    const { lastFrame, stdin, unmount } = render(<App repoRoot={EMPTY_ROOT} />);
    stdin.write('\r'); // enter on "New experiment"
    await new Promise((r) => setTimeout(r, 20));
    const frame = lastFrame() ?? '';
    expect(frame).toContain('New experiment');
    expect(frame).toContain('gemini');
    expect(frame).toContain('ollama');
    unmount();
  });
});
