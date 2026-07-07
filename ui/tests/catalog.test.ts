import { describe, expect, it } from 'vitest';
import { fetchOllamaCatalog, OLLAMA_TAGS_URL } from '../src/core/catalog';

const ok = (body: unknown): typeof fetch =>
  (async () =>
    new Response(JSON.stringify(body), { status: 200 })) as typeof fetch;

describe('fetchOllamaCatalog', () => {
  it('returns sorted unique catalog names from the tags schema', async () => {
    const names = await fetchOllamaCatalog(
      ok({
        models: [
          { name: 'glm-5.2' },
          { name: 'gemma4:31b' },
          { name: 'gemma4:31b' },
        ],
      }),
    );
    expect(names).toEqual(['gemma4:31b', 'glm-5.2']);
  });

  it('hits the documented endpoint', async () => {
    let url = '';
    await fetchOllamaCatalog((async (input: Parameters<typeof fetch>[0]) => {
      url = String(input);
      return new Response(JSON.stringify({ models: [{ name: 'x' }] }));
    }) as typeof fetch);
    expect(url).toBe(OLLAMA_TAGS_URL);
  });

  it('rejects on network failure (caller falls back to free text)', async () => {
    const failing = (async () => {
      throw new Error('network down');
    }) as unknown as typeof fetch;
    await expect(fetchOllamaCatalog(failing)).rejects.toThrow('network down');
  });

  it('rejects on non-2xx responses', async () => {
    const notFound = (async () =>
      new Response('nope', { status: 503 })) as typeof fetch;
    await expect(fetchOllamaCatalog(notFound)).rejects.toThrow('HTTP 503');
  });

  it('rejects on an empty catalog', async () => {
    await expect(fetchOllamaCatalog(ok({ models: [] }))).rejects.toThrow(
      'no models',
    );
  });
});
