# Benchmark Comparison & Recommendation — Harder Research-QA Dataset (Issue #42)

> **Correction (2026-06-07), found during integration — supersedes the license/tricky-tier cells below:**
> 1. **CREPE name collision.** The deep-research run cited `huggingface.co/datasets/zharry29/CREPE` as the false-presupposition CREPE. That HF dataset is a *different* CREPE ("Causal Reasoning of Entities and Events in Procedural Texts"). The real false-presupposition CREPE is arXiv:2211.17257 (`github.com/velocityCavalry/CREPE`), so the "CC-BY-4.0 on HF" claim was a **false match**.
> 2. **Licenses are murkier than the table implies.** FalseQA has **no declared license** (no LICENSE file / no README mention); the real CREPE is **NOASSERTION** (no detectable license). The clean BSD/CC-BY tags found belonged to re-hosts or the wrong dataset.
> 3. **Resolution.** We **do not redistribute** any external data — a fetch-at-build loader pulls subsets locally (git-ignored); the repo ships only mappers + an author-owned synthetic test fixture. This dissolves the license blocker for all candidates. **Integrated: FRAMES (hard, Apache-2.0) + CREPE (tricky, fetch-only).** See `docs/hard_benchmark.md`.


**Date:** 2026-06-05
**Method:** Multi-agent deep-research sweep (6 search angles, 23 sources fetched, 110 claims extracted, 25 adversarially fact-checked at 3 votes each — 22 confirmed, 3 killed). All figures below survived verification; refuted claims are explicitly flagged.

## Problem recap

The current `research_qa` pool is one-shot-saturated: a strong answerer (Gemini-class) maxes the quality judge at 5/5/5, the simulated user ends every episode after one turn, and conditions can't separate (see RCA in #41, and the no-signal baseline in #23). We need datasets where a frontier answerer scores **below ceiling with real variance**, that ship **gold answers + gradable rubrics** (so the LLM judge and follow-up simulated user keep working), and that map onto `ResearchQATask` (`question` / `gold_answer` / `scoring_rubric` / `difficulty`, with `tricky` = false-premise carrying `known_assumptions`, `hard` = multi-step synthesis).

## Comparison table

| Benchmark | 1. Difficulty / SOTA headroom | 2. Gold + rubric? | 3. License | 4. Multi-turn / multi-hop / agentic | 5. Loading | 6. False-premise? | Schema fit |
|---|---|---|---|---|---|---|---|
| **ResearchRubrics** | **High.** No system >70% rubric compliance; best Gemini DR 0.677 ternary / 0.615 binary; OpenAI DR 0.664; Perplexity DR 0.566. | **Yes — native.** 2,593 human-written rubric criteria (~26/task, 20–43), ternary judge {Satisfied / Partial / Not}. | ⚠️ Unresolved — verify before use | Long-horizon: 4+ dependent reasoning steps, synthesis across >5 sources. **Single-turn** (101 prompts). | ⚠️ HF availability unresolved | No | **Best (hard tier).** Native `scoring_rubric`; but scores measured on *agentic DR systems*, not a plain one-shot answerer. |
| **ResearchQA** | **High.** No parametric/RAG system >70% rubric-item coverage; best overall 75% (Perplexity DR). Top system fully addresses <11% citation, 48% limitation, 49% comparison items. | **Yes — native.** Query-specific free-text rubric items (cite-papers / explain / limitations); 160K rubric items, 21K queries, 75 fields. Pairwise auto-judge 74% agreement w/ experts. | ⚠️ Unresolved — verify | Long-form synthesis; **single-turn**. | ⚠️ HF availability unresolved | No | **Best (hard tier).** Cleanest native rubric mapping. |
| **FRAMES** | **High.** 0.408 no-retrieval (Gemini-1.5-Pro); 0.66 with multi-step retrieval; oracle-prompt only 0.73. Not saturated. | Short verifiable answers + reasoning types; needs a rubric authored. | ✅ on HF (`google/frames-benchmark`) | **Multi-hop** (2–15 Wikipedia docs/Q); rewards iterative retrieval. "Multi-step" = automated query iteration, not dialogue. | ✅ Easy — HF, 824 Qs. | No | **Good (hard injector).** Needs `gold_answer`→rubric authoring; multi-hop is the real headroom source. |
| **FalseQA** | High for premise-correction (frontier <30% on false-premise correction, corroborating sources). | Explanation of *why* premise is false + revised true-premise question. | ✅ Public GitHub (thunlp), ACL'23. | Single-turn; tests assumption-trap handling. | ✅ CSVs, 2,365 Qs (1187/491/687). | **Yes — purpose-built.** | **Best (tricky tier).** `question`→Q, `gold_answer`→rebuttal/explanation, `known_assumptions`→false premise. Minor reshape: TPQ is a sibling row, not a field. |
| **(QA)²** | High. Models underperform on questionable- vs valid-assumption Qs; substantial headroom (2026 corroboration: no frontier LLM corrects false presup >30%). | Annotated assumptions → convertible to Yes/No verification Qs. | ✅ ACL'23 (arXiv 2212.10003). | Single-turn. | ✅ 602 expert-annotated Qs (301 questionable: 246 false, 55 unverifiable). | **Yes.** | **Strong (tricky alt).** Uses human-rater acceptability, not hard auto-accuracy. |
| **CREPE** | Below ceiling: best ~66–67% macro-F1 vs 85% human (~18pt gap) on judging presupposition correctness. | Presuppositions **+ their corrections** annotated. | ✅ on HF (`zharry29/CREPE`). | Single-turn. | ✅ HF. 25% of Qs have false presuppositions. | **Yes — natural distribution** (from forums). | **Strong (tricky alt).** `known_assumptions`+`gold_answer` correction map directly. Baselines are 2022-era. |
| **BrowseComp** | Extreme: GPT-4o 0.6%, +browsing 1.9%, o1 9.9%; only Deep Research agent 51.5%. | Short single verifiable string, AI semantic-equivalence grading — **not rubric prose**. | OpenAI release. | Agentic browsing; **single-turn**; "persistent multi-hop navigation" claim **REFUTED**. | Loadable. | No | **Poor for this loop.** Near-zero is tool-access artifact, not reasoning; short answers, not gradable prose. Fits a tool-use injector, not rubric+follow-up. |
| **GAIA** | Hard agentic (reasoning/multimodal/browse/tool-use). The popular "92% human vs 15% GPT-4" headroom framing was **REFUTED 0-3** — don't cite it. | Short exact-match-style answers — **not free-text rubrics**. | Public. | Agentic tool-chaining; **single-turn**. | Loadable. | No | **Poor for this loop.** Difficulty injector / tool-use role only. |

## Recommendation

Adopt a **two-dataset pairing**, mirroring the existing `hard` / `tricky` difficulty split:

### Hard tier → **FRAMES** (primary integration target), with **ResearchQA / ResearchRubrics** as the rubric-native upgrade
- **FRAMES** is the pragmatic first integration: it's on HuggingFace (`google/frames-benchmark`, 824 Qs), genuinely multi-hop (2–15 docs/question), and frontier models land in the **0.40–0.73** band — real headroom, not saturation. The only schema work is authoring a `scoring_rubric` per question (the dataset gives gold answers + reasoning types to template from).
- **ResearchQA / ResearchRubrics** are the *cleanest schema fit* because they ship per-question free-text rubrics natively (no rubric authoring) and document <70% ceiling. **But** two caveats gate them: (a) their <70% scores were measured on *agentic deep-research systems*, not a plain one-shot answerer — below-ceiling behavior for our specific answerer is plausible but unproven; (b) license + HF-loadability were **not resolved** by the research and must be checked first. Treat these as the higher-fidelity follow-up once a pilot confirms headroom and licensing.

### Tricky tier → **FalseQA** (primary), with **CREPE** as a strong alternative
- **FalseQA** maps onto our schema most cleanly: `question`→Q, `gold_answer`→the false-premise rebuttal/explanation, `known_assumptions`→the false premise itself. It's purpose-built (2,365 human-written false-premise Qs), public, and CSV-simple. Minor reshape: the true-premise variant is a sibling row, not a column.
- **CREPE** is the better choice if we want a *natural* distribution of presupposition failures (from real forum questions) with corrections already annotated, and it's directly on HuggingFace.

## Critical caveats (read before integrating)

1. **Time-sensitivity is the #1 risk.** Nearly all "frontier struggles" numbers come from 2024–2025 models (Gemini-1.5-Pro on FRAMES, text-davinci-003 on (QA)², GPT-3 on CREPE). A 2026 Gemini-class answerer may score higher and compress the headroom. **The pilot (AC #3) must empirically confirm below-ceiling-with-variance on *our* answerer before committing.**
2. **Schema-fit vs difficulty tension.** The cleanest-rubric sets (ResearchQA/ResearchRubrics) are single-turn and were scored on agentic systems; the highest-headroom multi-hop set (FRAMES) needs rubric authoring. There's no single dataset that is simultaneously rubric-native, proven-hard-for-a-one-shot-answerer, and multi-turn.
3. **None are natively multi-turn.** The simulated-user follow-up loop must be layered on top. Rubric-graded long-form sets (ResearchRubrics/ResearchQA/FRAMES-with-rubric) support this far better than short-answer agentic sets — follow-ups can target unaddressed rubric items / limitations rather than being trivially satisfiable after turn one.
4. **Do NOT cite these refuted claims:** GAIA "92% human vs 15% GPT-4" headroom (0-3); BrowseComp "persistent multi-hop navigation where brute-force fails" (0-3); FRAMES "0.66 leaves a third unsolved / real below-ceiling variance" phrasing (0-3). The underlying FRAMES 0.40/0.66 numbers themselves are confirmed — only the editorialized framing was killed.

## Open questions to resolve during integration

1. Do 2026 frontier answerers actually score below ceiling on FRAMES / FalseQA / (QA)² / CREPE **inside this harness with tool access**, or has model progress already eaten the 2024-era headroom?
2. Can multi-criterion rubrics (ResearchRubrics ~26/task) collapse into the single `scoring_rubric: str` field without losing the granularity that produces score variance — or does the judge need to become multi-criterion rather than emitting one 5/5/5?
3. Are ResearchQA / ResearchRubrics redistributable and HF-loadable? (Unresolved — blocks their adoption.)
4. What follow-up design makes the simulated user create genuine multi-turn headroom (e.g., probing missing rubric items) rather than accepting turn one?

## Sources (primary unless noted)

- ResearchRubrics — arXiv:2511.07685v1 (Scale AI / ICLR 2026)
- ResearchQA — arXiv:2509.00496
- FRAMES — arXiv:2409.12941 (Google/Harvard, NAACL 2025); HF `google/frames-benchmark`
- FalseQA — arXiv:2307.02394, github.com/thunlp/FalseQA (ACL 2023)
- (QA)² — arXiv:2212.10003 / aclanthology 2023.acl-long.472 (ACL 2023)
- CREPE — arXiv:2211.17257 (UW+AI2, ACL 2023); HF `zharry29/CREPE`
- BrowseComp — arXiv:2504.12516v1 (OpenAI) + anthropic.com/engineering/eval-awareness-browsecomp (secondary)
- GAIA — arXiv:2311.12983
- Frontier false-premise corroboration (2026) — Cancer-Myth arXiv:2504.11373
