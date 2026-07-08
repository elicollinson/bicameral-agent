# Comparative Evaluation Report

- Dataset: `builtin` (metric: `llm_judge`)
- Answerer: `ollama/gemma4:31b-cloud`
- Measurement: `ollama/gemma4:31b-cloud`
- Tasks per condition: 100 (mix: {'typical': 50, 'hard': 25, 'tricky': 25})
- Max turns: 6; base seed: 42

Cells are `mean [95% CI] (n)`; n counts episodes where the metric
is defined (see module docs for the undefined cases).

## Summary

| Metric | no_subconscious | random | heuristic | learned_no_search | learned_with_search |
|---|---|---|---|---|---|
| task_quality | 0.6067 [0.5699, 0.6435] (n=100) | 0.6183 [0.5815, 0.6551] (n=100) | 0.6225 [0.5854, 0.6596] (n=100) | 0.6475 [0.6077, 0.6873] (n=100) | 0.6392 [0.6003, 0.6781] (n=100) |
| task_completed | 0.99 [0.9702, 1.01] (n=100) | 0.98 [0.9521, 1.008] (n=100) | 0.99 [0.9702, 1.01] (n=100) | 1 [1, 1] (n=100) | 0.99 [0.9702, 1.01] (n=100) |
| token_efficiency (lower=better) | 3330 [3139, 3520] (n=100) | 4057 [3731, 4383] (n=100) | 5516 [5381, 5651] (n=100) | 5465 [5338, 5591] (n=100) | 5680 [5463, 5897] (n=100) |
| user_stops (lower=better) | 0.01 [-0.009843, 0.02984] (n=100) | 0.02 [-0.00792, 0.04792] (n=100) | 0.01 [-0.009843, 0.02984] (n=100) | 0 [0, 0] (n=100) | 0.01 [-0.009843, 0.02984] (n=100) |
| time_to_completion_ms (lower=better) | 4.467e+04 [3.947e+04, 4.987e+04] (n=100) | 5.6e+04 [5.035e+04, 6.165e+04] (n=100) | 6.949e+04 [6.437e+04, 7.461e+04] (n=100) | 6.848e+04 [6.276e+04, 7.421e+04] (n=100) | 6.926e+04 [6.283e+04, 7.568e+04] (n=100) |
| tool_precision | - | 0.1 [-0.01396, 0.214] (n=30) | 0.41 [0.3119, 0.5081] (n=100) | 0.35 [0.2549, 0.4451] (n=100) | 0.34 [0.2455, 0.4345] (n=100) |
| interrupt_rate (lower=better) | 0 [0, 0] (n=100) | 0 [0, 0] (n=100) | 0 [0, 0] (n=100) | 0 [0, 0] (n=100) | 0 [0, 0] (n=100) |
| queue_expiry_rate (lower=better) | - | 0 [0, 0] (n=3) | 0 [0, 0] (n=41) | 0 [0, 0] (n=35) | 0 [0, 0] (n=34) |
| latency_mape (lower=better) | - | 47.55 [37.21, 57.9] (n=30) | 34.44 [29.65, 39.24] (n=100) | 37.17 [31.85, 42.49] (n=100) | 34.64 [29.51, 39.77] (n=100) |

## Pairwise Welch t-tests: task_quality

| A | B | mean A | mean B | t | p | significant |
|---|---|---|---|---|---|---|
| no_subconscious | random | 0.6067 | 0.6183 | -0.4449 | 0.6569 | no |
| no_subconscious | heuristic | 0.6067 | 0.6225 | -0.6011 | 0.5485 | no |
| no_subconscious | learned_no_search | 0.6067 | 0.6475 | -1.495 | 0.1366 | no |
| no_subconscious | learned_with_search | 0.6067 | 0.6392 | -1.204 | 0.2299 | no |
| random | heuristic | 0.6183 | 0.6225 | -0.1582 | 0.8745 | no |
| random | learned_no_search | 0.6183 | 0.6475 | -1.068 | 0.287 | no |
| random | learned_with_search | 0.6183 | 0.6392 | -0.7719 | 0.4411 | no |
| heuristic | learned_no_search | 0.6225 | 0.6475 | -0.9114 | 0.3632 | no |
| heuristic | learned_with_search | 0.6225 | 0.6392 | -0.615 | 0.5393 | no |
| learned_no_search | learned_with_search | 0.6475 | 0.6392 | 0.2971 | 0.7667 | no |

## Significant differences on other metrics (Welch 95%)

- token_efficiency: no_subconscious vs random (means 3330 vs 4057, t=-3.818, p=0.0001923)
- token_efficiency: no_subconscious vs heuristic (means 3330 vs 5516, t=-18.58, p=1.448e-43)
- token_efficiency: no_subconscious vs learned_no_search (means 3330 vs 5465, t=-18.51, p=8.107e-43)
- token_efficiency: no_subconscious vs learned_with_search (means 3330 vs 5680, t=-16.15, p=8.419e-38)
- token_efficiency: random vs heuristic (means 4057 vs 5516, t=-8.197, p=1.897e-13)
- token_efficiency: random vs learned_no_search (means 4057 vs 5465, t=-7.978, p=7.222e-13)
- token_efficiency: random vs learned_with_search (means 4057 vs 5680, t=-8.218, p=4.794e-14)
- time_to_completion_ms: no_subconscious vs random (means 4.467e+04 vs 5.6e+04, t=-2.929, p=0.003808)
- time_to_completion_ms: no_subconscious vs heuristic (means 4.467e+04 vs 6.949e+04, t=-6.749, p=1.609e-10)
- time_to_completion_ms: no_subconscious vs learned_no_search (means 4.467e+04 vs 6.848e+04, t=-6.108, p=5.321e-09)
- time_to_completion_ms: no_subconscious vs learned_with_search (means 4.467e+04 vs 6.926e+04, t=-5.902, p=1.623e-08)
- time_to_completion_ms: random vs heuristic (means 5.6e+04 vs 6.949e+04, t=-3.51, p=0.0005555)
- time_to_completion_ms: random vs learned_no_search (means 5.6e+04 vs 6.848e+04, t=-3.078, p=0.002378)
- time_to_completion_ms: random vs learned_with_search (means 5.6e+04 vs 6.926e+04, t=-3.074, p=0.002414)
- tool_precision: random vs heuristic (means 0.1 vs 0.41, t=-4.162, p=8.013e-05)
- tool_precision: random vs learned_no_search (means 0.1 vs 0.35, t=-3.402, p=0.001073)
- tool_precision: random vs learned_with_search (means 0.1 vs 0.34, t=-3.275, p=0.0016)
- latency_mape: random vs heuristic (means 47.55 vs 34.44, t=2.34, p=0.02396)
- latency_mape: random vs learned_with_search (means 47.55 vs 34.64, t=2.275, p=0.0277)

## Breakdown by difficulty

### typical

| Metric | no_subconscious | random | heuristic | learned_no_search | learned_with_search |
|---|---|---|---|---|---|
| task_quality | 0.5983 [0.5508, 0.6459] (n=50) | 0.625 [0.5761, 0.6739] (n=50) | 0.6267 [0.5797, 0.6736] (n=50) | 0.6417 [0.5873, 0.6961] (n=50) | 0.6433 [0.5918, 0.6949] (n=50) |
| task_completed | 1 [1, 1] (n=50) | 1 [1, 1] (n=50) | 1 [1, 1] (n=50) | 1 [1, 1] (n=50) | 1 [1, 1] (n=50) |
| token_efficiency (lower=better) | 3182 [2976, 3388] (n=50) | 3881 [3479, 4282] (n=50) | 5383 [5259, 5506] (n=50) | 5412 [5300, 5525] (n=50) | 5600 [5289, 5912] (n=50) |
| user_stops (lower=better) | 0 [0, 0] (n=50) | 0 [0, 0] (n=50) | 0 [0, 0] (n=50) | 0 [0, 0] (n=50) | 0 [0, 0] (n=50) |
| time_to_completion_ms (lower=better) | 4.736e+04 [3.949e+04, 5.523e+04] (n=50) | 5.121e+04 [4.406e+04, 5.836e+04] (n=50) | 7.142e+04 [6.368e+04, 7.916e+04] (n=50) | 7.029e+04 [6.235e+04, 7.824e+04] (n=50) | 7.637e+04 [6.737e+04, 8.537e+04] (n=50) |
| tool_precision | - | 0.1333 [-0.06203, 0.3287] (n=15) | 0.4 [0.2593, 0.5407] (n=50) | 0.42 [0.2783, 0.5617] (n=50) | 0.36 [0.2222, 0.4978] (n=50) |
| interrupt_rate (lower=better) | 0 [0, 0] (n=50) | 0 [0, 0] (n=50) | 0 [0, 0] (n=50) | 0 [0, 0] (n=50) | 0 [0, 0] (n=50) |
| queue_expiry_rate (lower=better) | - | 0 [0, 0] (n=2) | 0 [0, 0] (n=20) | 0 [0, 0] (n=21) | 0 [0, 0] (n=18) |
| latency_mape (lower=better) | - | 45.27 [29.04, 61.49] (n=15) | 31.78 [25.11, 38.45] (n=50) | 36.69 [28.36, 45.02] (n=50) | 39.47 [31.96, 46.99] (n=50) |

### hard

| Metric | no_subconscious | random | heuristic | learned_no_search | learned_with_search |
|---|---|---|---|---|---|
| task_quality | 0.54 [0.4795, 0.6005] (n=25) | 0.5133 [0.4671, 0.5596] (n=25) | 0.55 [0.5186, 0.5814] (n=25) | 0.53 [0.4657, 0.5943] (n=25) | 0.5433 [0.4862, 0.6005] (n=25) |
| task_completed | 1 [1, 1] (n=25) | 1 [1, 1] (n=25) | 1 [1, 1] (n=25) | 1 [1, 1] (n=25) | 1 [1, 1] (n=25) |
| token_efficiency (lower=better) | 3748 [3494, 4003] (n=25) | 4451 [3896, 5005] (n=25) | 6162 [5935, 6389] (n=25) | 6049 [5897, 6201] (n=25) | 6097 [5939, 6255] (n=25) |
| user_stops (lower=better) | 0 [0, 0] (n=25) | 0 [0, 0] (n=25) | 0 [0, 0] (n=25) | 0 [0, 0] (n=25) | 0 [0, 0] (n=25) |
| time_to_completion_ms (lower=better) | 3.952e+04 [3.279e+04, 4.625e+04] (n=25) | 7.082e+04 [5.917e+04, 8.248e+04] (n=25) | 6.176e+04 [5.54e+04, 6.813e+04] (n=25) | 6.148e+04 [5.41e+04, 6.887e+04] (n=25) | 6.475e+04 [5.655e+04, 7.295e+04] (n=25) |
| tool_precision | - | 0 [0, 0] (n=9) | 0.2 [0.03138, 0.3686] (n=25) | 0.16 [0.005455, 0.3145] (n=25) | 0.12 [-0.01699, 0.257] (n=25) |
| interrupt_rate (lower=better) | 0 [0, 0] (n=25) | 0 [0, 0] (n=25) | 0 [0, 0] (n=25) | 0 [0, 0] (n=25) | 0 [0, 0] (n=25) |
| queue_expiry_rate (lower=better) | - | - | 0 [0, 0] (n=5) | 0 [0, 0] (n=4) | 0 [0, 0] (n=3) |
| latency_mape (lower=better) | - | 59.22 [44.23, 74.2] (n=9) | 35.05 [26.12, 43.98] (n=25) | 38.61 [27.57, 49.65] (n=25) | 28.48 [19.07, 37.89] (n=25) |

### tricky

| Metric | no_subconscious | random | heuristic | learned_no_search | learned_with_search |
|---|---|---|---|---|---|
| task_quality | 0.69 [0.5965, 0.7835] (n=25) | 0.71 [0.6189, 0.8011] (n=25) | 0.6867 [0.575, 0.7983] (n=25) | 0.7767 [0.6992, 0.8541] (n=25) | 0.7267 [0.6301, 0.8233] (n=25) |
| task_completed | 0.96 [0.8774, 1.043] (n=25) | 0.92 [0.8056, 1.034] (n=25) | 0.96 [0.8774, 1.043] (n=25) | 1 [1, 1] (n=25) | 0.96 [0.8774, 1.043] (n=25) |
| token_efficiency (lower=better) | 3208 [2608, 3808] (n=25) | 4016 [3087, 4945] (n=25) | 5138 [4817, 5458] (n=25) | 4985 [4659, 5312] (n=25) | 5423 [4829, 6016] (n=25) |
| user_stops (lower=better) | 0.04 [-0.04261, 0.1226] (n=25) | 0.08 [-0.03437, 0.1944] (n=25) | 0.04 [-0.04261, 0.1226] (n=25) | 0 [0, 0] (n=25) | 0.04 [-0.04261, 0.1226] (n=25) |
| time_to_completion_ms (lower=better) | 4.444e+04 [3.162e+04, 5.726e+04] (n=25) | 5.078e+04 [3.804e+04, 6.352e+04] (n=25) | 7.336e+04 [6.091e+04, 8.581e+04] (n=25) | 7.187e+04 [5.612e+04, 8.761e+04] (n=25) | 5.955e+04 [4.272e+04, 7.638e+04] (n=25) |
| tool_precision | - | 0.1667 [-0.2618, 0.5952] (n=6) | 0.64 [0.4377, 0.8423] (n=25) | 0.4 [0.1935, 0.6065] (n=25) | 0.52 [0.3094, 0.7306] (n=25) |
| interrupt_rate (lower=better) | 0 [0, 0] (n=25) | 0 [0, 0] (n=25) | 0 [0, 0] (n=25) | 0 [0, 0] (n=25) | 0 [0, 0] (n=25) |
| queue_expiry_rate (lower=better) | - | 0 [0, 0] (n=1) | 0 [0, 0] (n=16) | 0 [0, 0] (n=10) | 0 [0, 0] (n=13) |
| latency_mape (lower=better) | - | 35.78 [2.238, 69.32] (n=6) | 39.15 [27.62, 50.67] (n=25) | 36.69 [27.46, 45.92] (n=25) | 31.13 [20.13, 42.12] (n=25) |

## Transport failures

- none

## Reproducibility

Episode collection is LLM-stochastic; task selection/pairing,
controller seeds, and everything downstream of the collected
episodes (metrics, CIs, tests, this report) are deterministic
for the recorded base seed.
