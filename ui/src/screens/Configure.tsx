import React, { useEffect, useState } from 'react';
import { Box, Text, useInput } from 'ink';
import Field from '../components/Field';
import SearchSelect from '../components/SearchSelect';
import { fetchOllamaCatalog } from '../core/catalog';
import {
  buildRunnerArgs,
  defaultOutputDir,
  formatCommand,
  PROVIDERS,
  type ExperimentConfig,
  type Provider,
} from '../core/command';
import { geminiModels, hasPricing } from '../core/pricing';

type Step =
  | 'provider'
  | 'model'
  | 'tasks'
  | 'maxTurns'
  | 'parallel'
  | 'budget'
  | 'outputDir'
  | 'confirm';

const STEP_ORDER: Step[] = [
  'provider',
  'model',
  'tasks',
  'maxTurns',
  'parallel',
  'budget',
  'outputDir',
  'confirm',
];

type CatalogState =
  | { kind: 'loading' }
  | { kind: 'ready'; models: string[] }
  | { kind: 'failed'; reason: string };

interface Props {
  pricingKeys: string[];
  onLaunch: (config: ExperimentConfig) => void;
  onCancel: () => void;
}

const positiveInt = (value: string): string | null =>
  /^\d+$/.test(value) && Number(value) > 0
    ? null
    : 'enter a positive whole number';

export default function Configure({ pricingKeys, onLaunch, onCancel }: Props) {
  const [step, setStep] = useState<Step>('provider');
  const [provider, setProvider] = useState<Provider>('gemini');
  const [providerIndex, setProviderIndex] = useState(0);
  const [model, setModel] = useState('');
  const [manualModel, setManualModel] = useState(false);
  const [catalog, setCatalog] = useState<CatalogState>({ kind: 'loading' });
  const [tasks, setTasks] = useState('5');
  const [maxTurns, setMaxTurns] = useState('10');
  const [parallel, setParallel] = useState('1');
  const [budget, setBudget] = useState('');
  const [outputDir, setOutputDir] = useState(defaultOutputDir());

  const back = () => {
    const i = STEP_ORDER.indexOf(step);
    if (i === 0) onCancel();
    else setStep(STEP_ORDER[i - 1]);
  };
  const next = () => setStep(STEP_ORDER[STEP_ORDER.indexOf(step) + 1]);

  // Fetch the Ollama Cloud catalog when the model step needs it.
  useEffect(() => {
    if (step !== 'model' || provider !== 'ollama') return;
    let cancelled = false;
    setCatalog({ kind: 'loading' });
    fetchOllamaCatalog()
      .then((models) => {
        if (!cancelled) setCatalog({ kind: 'ready', models });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setCatalog({ kind: 'failed', reason: String(err) });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [step, provider]);

  const config: ExperimentConfig = {
    provider,
    model,
    tasksPerCondition: Number(tasks),
    maxTurns: Number(maxTurns),
    parallelEpisodes: Number(parallel),
    episodeBudget: budget === '' ? null : Number(budget),
    outputDir,
    configPath: null,
  };

  useInput(
    (_input, key) => {
      if (step !== 'provider') return;
      if (key.escape) onCancel();
      else if (key.upArrow) setProviderIndex((i) => Math.max(i - 1, 0));
      else if (key.downArrow) {
        setProviderIndex((i) => Math.min(i + 1, PROVIDERS.length - 1));
      } else if (key.return) {
        setProvider(PROVIDERS[providerIndex]);
        setManualModel(false);
        setModel('');
        next();
      }
    },
    { isActive: step === 'provider' },
  );

  useInput(
    (_input, key) => {
      if (key.escape) back();
      else if (key.return) onLaunch(config);
    },
    { isActive: step === 'confirm' },
  );

  const pickModel = (name: string) => {
    setModel(name);
    setStep('tasks');
  };

  // Each Field is keyed by its step: consecutive steps render <Field> at the
  // same tree position, and without a key React reuses the instance, carrying
  // the previous step's text into the next one.
  const modelField = (
    <Field
      key="model"
      label="Model tag"
      initialValue={model}
      hint={
        provider === 'ollama'
          ? 'Catalog name exactly as on ollama.com, e.g. gemma4:31b'
          : 'Gemini model id, e.g. gemini-3.1-flash-lite-preview'
      }
      validate={(v) => (v.trim() === '' ? 'model tag is required' : null)}
      onSubmit={(v) => pickModel(v.trim())}
      onCancel={() => {
        if (manualModel) setManualModel(false);
        else back();
      }}
    />
  );

  let body: React.ReactNode;
  switch (step) {
    case 'provider':
      body = (
        <Box flexDirection="column">
          <Text>Provider:</Text>
          {PROVIDERS.map((p, i) => (
            <Text key={p} color={i === providerIndex ? 'green' : undefined}>
              {i === providerIndex ? '❯ ' : '  '}
              {p}
            </Text>
          ))}
          <Text dimColor>enter select · ↑/↓ move · esc menu</Text>
        </Box>
      );
      break;
    case 'model': {
      if (manualModel) {
        body = modelField;
      } else if (provider === 'gemini') {
        const models = geminiModels(pricingKeys);
        body =
          models.length > 0 ? (
            <Box flexDirection="column">
              <Text>Model (from MODEL_PRICING):</Text>
              <SearchSelect
                items={models}
                onSelect={pickModel}
                onManual={() => setManualModel(true)}
                onCancel={back}
              />
            </Box>
          ) : (
            modelField
          );
      } else if (catalog.kind === 'loading') {
        body = <Text>Fetching Ollama Cloud catalog…</Text>;
      } else if (catalog.kind === 'ready') {
        body = (
          <Box flexDirection="column">
            <Text>Model (Ollama Cloud catalog):</Text>
            <SearchSelect
              items={catalog.models}
              onSelect={pickModel}
              onManual={() => setManualModel(true)}
              onCancel={back}
            />
          </Box>
        );
      } else {
        body = (
          <Box flexDirection="column">
            <Text color="yellow">
              Catalog unavailable ({catalog.reason}) — enter the model tag
              manually.
            </Text>
            {modelField}
          </Box>
        );
      }
      break;
    }
    case 'tasks':
      body = (
        <Field
          key="tasks"
          label="Tasks per condition"
          initialValue={tasks}
          validate={positiveInt}
          onSubmit={(v) => {
            setTasks(v);
            next();
          }}
          onCancel={back}
        />
      );
      break;
    case 'maxTurns':
      body = (
        <Field
          key="maxTurns"
          label="Max turns per episode"
          initialValue={maxTurns}
          validate={positiveInt}
          onSubmit={(v) => {
            setMaxTurns(v);
            next();
          }}
          onCancel={back}
        />
      );
      break;
    case 'parallel':
      body = (
        <Field
          key="parallel"
          label="Parallel episodes"
          initialValue={parallel}
          hint="Match the provider's concurrent-request allowance; 1 = sequential"
          validate={positiveInt}
          onSubmit={(v) => {
            setParallel(v);
            next();
          }}
          onCancel={back}
        />
      );
      break;
    case 'budget':
      body = (
        <Field
          key="budget"
          label="Per-episode budget (USD)"
          initialValue={budget}
          hint="Empty = no per-episode ceiling"
          validate={(v) =>
            v === '' || (Number.isFinite(Number(v)) && Number(v) > 0)
              ? null
              : 'enter a positive number or leave empty'
          }
          onSubmit={(v) => {
            setBudget(v.trim());
            next();
          }}
          onCancel={back}
        />
      );
      break;
    case 'outputDir':
      body = (
        <Field
          key="outputDir"
          label="Output dir (relative to repo root)"
          initialValue={outputDir}
          validate={(v) => (v.trim() === '' ? 'output dir is required' : null)}
          onSubmit={(v) => {
            setOutputDir(v.trim());
            next();
          }}
          onCancel={back}
        />
      );
      break;
    case 'confirm': {
      const priced = hasPricing(model, pricingKeys);
      body = (
        <Box flexDirection="column">
          <Text bold>Ready to launch</Text>
          <Text>
            {'  '}provider={provider} model={model} tasks={tasks} max-turns=
            {maxTurns} parallel={parallel} budget={budget || 'none'}
          </Text>
          <Text>
            {'  '}output: {outputDir}
          </Text>
          <Text dimColor>{formatCommand(buildRunnerArgs(config))}</Text>
          {!priced && (
            <Text color="yellow">
              Warning: '{model}' has no MODEL_PRICING entry — unregistered tags
              have crashed mid-run before (#52). Cost tracking may fail.
            </Text>
          )}
          <Text dimColor>enter launch · esc back</Text>
        </Box>
      );
      break;
    }
  }

  return (
    <Box flexDirection="column">
      <Text bold color="cyan">
        New experiment
      </Text>
      {body}
    </Box>
  );
}
