import fs from 'node:fs/promises';
import { parquetMetadata } from 'hyparquet';

/** Row count of a parquet file, or null if missing/unreadable/partial.
 *
 * The runner rewrites `<output-dir>/<condition>.parquet` after every
 * completed episode, so the row count is the live episode counter. A null
 * (e.g. read raced a rewrite) just skips one poll tick.
 */
export async function parquetRowCount(file: string): Promise<number | null> {
  try {
    const buf = await fs.readFile(file);
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    return Number(parquetMetadata(ab).num_rows);
  } catch {
    return null;
  }
}
