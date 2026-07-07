import React, { useMemo, useState } from 'react';
import { Box, Text, useInput } from 'ink';

const VISIBLE = 8;

interface Props {
  items: string[];
  onSelect: (item: string) => void;
  /** Called when the user presses tab to switch to free-text entry. */
  onManual?: () => void;
  onCancel?: () => void;
}

/** Type-to-filter list picker (used for model selection). */
export default function SearchSelect({
  items,
  onSelect,
  onManual,
  onCancel,
}: Props) {
  const [query, setQuery] = useState('');
  const [index, setIndex] = useState(0);

  const filtered = useMemo(() => {
    const q = query.toLowerCase();
    return items.filter((item) => item.toLowerCase().includes(q));
  }, [items, query]);

  const clamped = Math.min(index, Math.max(filtered.length - 1, 0));
  const windowStart = Math.max(
    0,
    Math.min(clamped - Math.floor(VISIBLE / 2), filtered.length - VISIBLE),
  );
  const visible = filtered.slice(windowStart, windowStart + VISIBLE);

  useInput((input, key) => {
    if (key.escape) {
      onCancel?.();
    } else if (key.tab) {
      onManual?.();
    } else if (key.upArrow) {
      setIndex(Math.max(clamped - 1, 0));
    } else if (key.downArrow) {
      setIndex(Math.min(clamped + 1, filtered.length - 1));
    } else if (key.return) {
      if (filtered[clamped] !== undefined) onSelect(filtered[clamped]);
    } else if (key.backspace || key.delete) {
      setQuery((q) => q.slice(0, -1));
      setIndex(0);
    } else if (input && !key.ctrl && !key.meta) {
      setQuery((q) => q + input);
      setIndex(0);
    }
  });

  return (
    <Box flexDirection="column">
      <Text>
        Filter: <Text color="cyan">{query || '(type to filter)'}</Text>{' '}
        <Text dimColor>
          {filtered.length}/{items.length} models
        </Text>
      </Text>
      {visible.map((item, i) => {
        const selected = windowStart + i === clamped;
        return (
          <Text key={item} color={selected ? 'green' : undefined}>
            {selected ? '❯ ' : '  '}
            {item}
          </Text>
        );
      })}
      {filtered.length === 0 && <Text dimColor>  (no matches)</Text>}
      <Text dimColor>
        enter select · tab manual entry · ↑/↓ move · esc back
      </Text>
    </Box>
  );
}
