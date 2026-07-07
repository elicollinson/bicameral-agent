/** MODEL_PRICING introspection.
 *
 * The registry lives in Python (src/bicameral_agent/cost_tracker.py). The UI
 * reads the keys straight out of that source file so the two never drift —
 * without importing Python or duplicating the list. Selecting a model with
 * no pricing entry is allowed but warned about: unregistered tags crash the
 * cost tracker mid-run (issue #52).
 */

import fs from 'node:fs';
import path from 'node:path';

export const COST_TRACKER_RELPATH = path.join(
  'src',
  'bicameral_agent',
  'cost_tracker.py',
);

/** Extract the model keys of the MODEL_PRICING dict from Python source. */
export function parsePricingKeys(source: string): string[] {
  const keys: string[] = [];
  const re = /"([^"]+)":\s*ModelPricing\(/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(source)) !== null) {
    keys.push(match[1]);
  }
  return keys;
}

/** Read pricing keys from the repo; empty list if the file is unreadable. */
export function loadPricingKeys(repoRoot: string): string[] {
  try {
    const source = fs.readFileSync(
      path.join(repoRoot, COST_TRACKER_RELPATH),
      'utf8',
    );
    return parsePricingKeys(source);
  } catch {
    return [];
  }
}

/** Whether a selected model tag has a pricing entry.
 *
 * Catalog names are checked as-is and with the `-cloud` suffix the Python
 * config currently registers for Ollama Cloud tags — the suffix is a
 * transport convention, not a different model, so either entry counts.
 */
export function hasPricing(model: string, pricingKeys: string[]): boolean {
  return pricingKeys.includes(model) || pricingKeys.includes(`${model}-cloud`);
}

export function geminiModels(pricingKeys: string[]): string[] {
  return pricingKeys.filter((k) => k.startsWith('gemini'));
}
