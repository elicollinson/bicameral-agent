import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import TextInput from 'ink-text-input';

interface Props {
  label: string;
  initialValue: string;
  hint?: string;
  /** Return an error message to reject, or null to accept. */
  validate?: (value: string) => string | null;
  onSubmit: (value: string) => void;
  onCancel?: () => void;
}

/** One wizard text field with submit-time validation. */
export default function Field({
  label,
  initialValue,
  hint,
  validate,
  onSubmit,
  onCancel,
}: Props) {
  const [value, setValue] = useState(initialValue);
  const [error, setError] = useState<string | null>(null);

  useInput((_input, key) => {
    if (key.escape) onCancel?.();
  });

  return (
    <Box flexDirection="column">
      <Box>
        <Text>{label}: </Text>
        <TextInput
          value={value}
          onChange={(v) => {
            setValue(v);
            setError(null);
          }}
          onSubmit={(v) => {
            const problem = validate?.(v) ?? null;
            if (problem) {
              setError(problem);
            } else {
              onSubmit(v);
            }
          }}
        />
      </Box>
      {hint && <Text dimColor>{hint}</Text>}
      {error && <Text color="red">{error}</Text>}
      <Text dimColor>enter accept · esc back</Text>
    </Box>
  );
}
