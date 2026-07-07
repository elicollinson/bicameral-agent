import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  geminiModels,
  hasPricing,
  loadPricingKeys,
  parsePricingKeys,
} from '../src/core/pricing';

const SAMPLE = `
MODEL_PRICING: dict[str, ModelPricing] = {
    "gemini-3.1-flash-lite-preview": ModelPricing(
        input_cost_per_token=0.50 / 1_000_000,
        output_cost_per_token=3.00 / 1_000_000,
    ),
    "gemma4:31b-cloud": ModelPricing(
        input_cost_per_token=0.0,
        output_cost_per_token=0.0,
    ),
}
`;

describe('parsePricingKeys', () => {
  it('extracts model keys from the MODEL_PRICING dict', () => {
    expect(parsePricingKeys(SAMPLE)).toEqual([
      'gemini-3.1-flash-lite-preview',
      'gemma4:31b-cloud',
    ]);
  });

  it('returns empty for unrelated source', () => {
    expect(parsePricingKeys('x = 1')).toEqual([]);
  });
});

describe('loadPricingKeys', () => {
  it('reads keys from the real repo cost_tracker.py', () => {
    const repoRoot = path.resolve(__dirname, '../..');
    const keys = loadPricingKeys(repoRoot);
    expect(keys).toContain('gemini-3.1-flash-lite-preview');
  });

  it('returns empty when the file is missing', () => {
    expect(loadPricingKeys('/nonexistent')).toEqual([]);
  });
});

describe('hasPricing', () => {
  const keys = ['gemini-3.1-flash-lite-preview', 'gemma4:31b-cloud'];

  it('matches exact keys', () => {
    expect(hasPricing('gemini-3.1-flash-lite-preview', keys)).toBe(true);
  });

  it('treats a catalog name as priced when its -cloud twin is registered', () => {
    expect(hasPricing('gemma4:31b', keys)).toBe(true);
  });

  it('flags unregistered tags', () => {
    expect(hasPricing('glm-5.2', keys)).toBe(false);
  });
});

describe('geminiModels', () => {
  it('filters to gemini keys only', () => {
    expect(
      geminiModels(['gemini-3.1-flash-lite-preview', 'gemma4:31b-cloud']),
    ).toEqual(['gemini-3.1-flash-lite-preview']);
  });
});
