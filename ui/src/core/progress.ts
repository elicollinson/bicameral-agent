/** Parse benchmark-runner output lines into live progress state.
 *
 * The runner announces each condition on stderr
 * ("Running N episodes for condition 'x'") and prints the final report to
 * stdout, from which per-condition mean cost is scraped. Anything that does
 * not match is ignored — the raw tail is shown to the user regardless.
 */

export interface ProgressState {
  /** Episodes per condition, once announced by the runner. */
  totalPerCondition: number | null;
  /** Condition currently being run, per the latest runner announcement. */
  activeCondition: string | null;
  /** Per-condition mean tool cost (USD) scraped from the final report. */
  costByCondition: Record<string, number>;
  /** Report section currently being parsed (internal cursor). */
  reportCondition: string | null;
}

export function initialProgress(): ProgressState {
  return {
    totalPerCondition: null,
    activeCondition: null,
    costByCondition: {},
    reportCondition: null,
  };
}

const RUNNING_RE = /Running (\d+) episodes for condition '([^']+)'/;
const REPORT_CONDITION_RE = /^## Condition: (\S+)/;
const COST_RE = /^\s*tool_cost_usd\s+mean=([0-9.eE+-]+)/;

/** Fold one output line (stdout or stderr) into the progress state. */
export function applyLine(state: ProgressState, line: string): ProgressState {
  const running = RUNNING_RE.exec(line);
  if (running) {
    return {
      ...state,
      totalPerCondition: Number(running[1]),
      activeCondition: running[2],
    };
  }
  const section = REPORT_CONDITION_RE.exec(line);
  if (section) {
    return { ...state, reportCondition: section[1] };
  }
  const cost = COST_RE.exec(line);
  if (cost && state.reportCondition) {
    const value = Number(cost[1]);
    if (Number.isFinite(value)) {
      return {
        ...state,
        costByCondition: {
          ...state.costByCondition,
          [state.reportCondition]: value,
        },
      };
    }
  }
  return state;
}
