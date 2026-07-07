import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { parquetRowCount } from '../src/core/parquet';

// A real artifact committed with the historical baseline run.
const BASELINE_PARQUET = path.resolve(
  __dirname,
  '../../data/baseline/no_subconscious.parquet',
);

describe('parquetRowCount', () => {
  it.skipIf(!fs.existsSync(BASELINE_PARQUET))(
    'reads the row count of a real runner artifact',
    async () => {
      expect(await parquetRowCount(BASELINE_PARQUET)).toBe(50);
    },
  );

  it('returns null for a missing file', async () => {
    expect(await parquetRowCount('/nonexistent.parquet')).toBeNull();
  });

  it('returns null for a torn/partial file', async () => {
    const tmp = path.join(__dirname, 'fixtures', 'torn.parquet.tmp');
    fs.writeFileSync(tmp, 'PAR1 not really parquet');
    try {
      expect(await parquetRowCount(tmp)).toBeNull();
    } finally {
      fs.unlinkSync(tmp);
    }
  });
});
