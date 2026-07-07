import fs from 'node:fs';
import path from 'node:path';

/** Walk up from `start` to the directory containing pyproject.toml.
 *
 * Lets the UI run from either the repo root or ui/. Falls back to `start`
 * if no marker is found (the app still renders; launching then fails
 * visibly with a spawn error).
 */
export function findRepoRoot(start: string = process.cwd()): string {
  let dir = path.resolve(start);
  for (;;) {
    if (fs.existsSync(path.join(dir, 'pyproject.toml'))) return dir;
    const parent = path.dirname(dir);
    if (parent === dir) return path.resolve(start);
    dir = parent;
  }
}
