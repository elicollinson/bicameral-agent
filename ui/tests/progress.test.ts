import { describe, expect, it } from 'vitest';
import { applyLine, initialProgress } from '../src/core/progress';

describe('applyLine', () => {
  it('tracks the announced condition and episode total', () => {
    const state = applyLine(
      initialProgress(),
      "2026-07-06 10:00:00,000 INFO Running 5 episodes for condition 'random'",
    );
    expect(state.totalPerCondition).toBe(5);
    expect(state.activeCondition).toBe('random');
  });

  it('scrapes per-condition mean cost from the final report', () => {
    let state = initialProgress();
    state = applyLine(state, '## Condition: heuristic  (n=50)');
    state = applyLine(
      state,
      '  tool_cost_usd            mean=0.006 std=0.001 95% CI=[0.005, 0.006] n=50',
    );
    expect(state.costByCondition).toEqual({ heuristic: 0.006 });
  });

  it('ignores cost lines before any report section', () => {
    const state = applyLine(initialProgress(), '  tool_cost_usd mean=0.006');
    expect(state.costByCondition).toEqual({});
  });

  it('leaves state unchanged for unrelated lines', () => {
    const start = initialProgress();
    expect(applyLine(start, 'some noise')).toBe(start);
  });
});
