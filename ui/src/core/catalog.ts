/** Ollama Cloud model discovery.
 *
 * GET https://ollama.com/api/tags returns the public catalog (standard
 * Ollama tags schema). Names are surfaced verbatim — the UI never appends
 * or strips a `-cloud` suffix; any such translation is the Python side's
 * concern (issue #58 naming decision).
 */

export const OLLAMA_TAGS_URL = 'https://ollama.com/api/tags';

export async function fetchOllamaCatalog(
  fetchFn: typeof fetch = fetch,
  timeoutMs = 5000,
): Promise<string[]> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetchFn(OLLAMA_TAGS_URL, { signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status} from ${OLLAMA_TAGS_URL}`);
    const body = (await res.json()) as { models?: Array<{ name?: unknown }> };
    const names = (body.models ?? [])
      .map((m) => m?.name)
      .filter((n): n is string => typeof n === 'string' && n.length > 0);
    if (names.length === 0) throw new Error('catalog response contained no models');
    return [...new Set(names)].sort();
  } finally {
    clearTimeout(timer);
  }
}
