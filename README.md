# satgen: multi-model Digital SAT question pipeline

[![ci](https://github.com/Dulkinsar/sat-question-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Dulkinsar/sat-question-pipeline/actions/workflows/ci.yml)

Generates Digital SAT questions with a cheap model, filters them with repeated solves, sends survivors
to a stronger model for review, confirms any answer-key change with a blind re-solve, deduplicates
against the existing bank, and exports JSON for bulk upload.

It rebuilds a pipeline I built in 2025 at nFactorial Academy; the original code belongs to the company.
This version was written in October 2026 with Claude Code (AI-assisted).
[docs/background.md](docs/background.md) covers the original.

```
generate    Gemini 2.5 Flash-Lite, JSON mode, curated style context
  -> shuffle    choice order randomised per question; the key moves with its text
  -> solve x5   solver answers without seeing the key
  -> gate       >= 3/5 votes match the key ............................ -> review
                key fails, but >= 3/5 agree on another answer (salvage) -> review as suspected key error
                otherwise ............................................... screened out
  -> review     Claude Sonnet solves it with key and votes hidden, judges the wording;
                the code compares its answer with the key
  -> re-check   a key change ships only if the review model, solving again, agrees 2 of 3 times
  -> dedupe     MinHash + LSH against this run and the existing bank
  -> export     accepted questions as JSON; disputed items listed for a human
```

## The reviewer

The reviewer returns three judgements: its own answer, whether the key matches it, and whether the
wording has exactly one defensible answer. An earlier version returned a single `accepted` flag, and
on real models it rejected every wrong-keyed question outright while naming the right answer, so
salvage almost never reached the re-check (3 of 85 corrections in that run). Splitting the verdict
turns a wrong key into a correction.

The default review is key-hidden (`--review-mode keyless`): the reviewer sees only the question and
the code compares answers. It needs no trust in a `key_correct` flag, and it cannot anchor on the key
or the votes. `--review-mode key` shows both, as the original did.

## Evaluation

`data/eval/eval_set.jsonl` holds 234 labelled items (`scripts/build_eval_set.py`): 83 `good`, the same
83 with the key moved to a distractor (`bad_key`), and 68 with a distractor replaced by a second
defensible answer (`ambiguous`). Verbal items are hand-written; math items come from templates with
distractors built from real slips. Each item is solved five times and reviewed three ways (key and votes
shown, key only, nothing shown), so every configuration is scored from the same calls.

Run on 15 October 2026, Claude Haiku 5.5 solving, Claude Sonnet 5.5 reviewing, threshold 3 of 5. Raw
per-item output is in [`eval/results/`](eval/results/).

| Configuration | Shipped | Error rate | Good kept | Bad keys caught | Ambiguous caught | Cost |
|---|---|---|---|---|---|---|
| Gate only | 135 | 0.415 | 0.952 | 0.988 | 0.191 | $0.06 |
| Review only, key shown | 165 | 0.006 | 1.0 | 1.0 | 0.985 | $1.42 |
| Review only, key hidden | 165 | 0.0 | 1.0 | 1.0 | 1.0 | $1.70 |
| Gate + review, key shown | 82 | 0.024 | 0.952 | 1.0 | 0.971 | $0.77 |
| Full, key shown | 163 | 0.012 | 0.952 | 1.0 | 0.971 | $1.45 |
| Full, key hidden (default) | 161 | 0.0 | 0.952 | 1.0 | 1.0 | $1.75 |

"Bad keys caught" means the item shipped with the right key or did not ship; "Shipped" includes
wrong-keyed items that shipped after a confirmed correction.

- **The gate alone misses ambiguity** (0.191 caught): five solvers can agree on one reading of a question
  that has two defensible answers. That gap is why the reviewer exists.
- **Salvage works once the verdict is split.** All 82 corrections the default reviewer acted on were
  right and confirmed by the re-check, so the full design ships 161 questions where gate + review ships 82.
- **The key-hidden reviewer made no shipping errors**; reviewers shown the key let one or two ambiguous
  items through. It costs about 20% more, because it always writes a full explanation.
- **Anchoring on the key is small once the verdict is split.** For each good/bad-key twin, the wording is
  identical, so a reviewer should judge ambiguity the same way. Shown the key, it flipped on 1 to 2 of 83
  pairs; key-hidden, where both twins get the same prompt, it flipped on 1 of 83, which is run-to-run
  noise. The 96% flip rate under the single-flag reviewer came from the flag, not from anchoring.
- **The gate saves money only without salvage.** With salvage, most items reach the reviewer anyway.

Limits: the planted errors are blunt, the math is templated, and Claude reviews questions a Claude
model solved, so these numbers show the harness separates the cases and say little about subtle errors
in fresh questions. After a parser fix (reviewers sometimes name an answer by its value, "8", or name
two answers when the wording is ambiguous), the 69 items with an unparsed review were re-run and merged;
their ids are in `eval/results/rerun_after_parser_fix.txt`. The next measurement is a real run on
generated questions with a human labelling a sample.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                       # no network needed
python -m satgen run --offline --count 36    # end-to-end run with simulated models
python -m satgen eval --simulate             # harness smoke test (scores are meaningless)
```

With real models (`pip install -e ".[live]"`, keys in `.env`):

```bash
python -m satgen run --count 200 --bank my_existing_bank.json
python -m satgen eval --solver-model claude-haiku-5-5
python -m satgen eval --rescore eval/results/<run>.items.jsonl   # re-score saved output, no API calls
```

A run writes each finished question to `out/checkpoint.jsonl` and resumes on rerun, retrying API
failures. The checkpoint records models, settings and a hash of the prompts and style context, and
refuses to resume under different ones. Outputs: `out/questions.json` (accepted questions) and
`out/report.json` (status counts, cost per model and stage, items for a human, near-duplicate pairs).
`examples/` has a sample offline run.

## Design notes

- **Statuses.** Every attempt ends as `generation_failed`, `error`, `screened_out`, `rejected`,
  `needs_human`, `duplicate` or `accepted`, so rates are over everything attempted.
- **Parsing.** Solvers answer in JSON; otherwise only the first line is read and anything unclear counts
  as `?`. Numeric answers handle Unicode minus, LaTeX fractions, thousands separators, a decimal comma and
  rounded decimals (0.333 matches 1/3).
- **Correlated errors.** By default the generator also solves, so one model's misconception can pass the
  gate; `--solver-model` swaps in another model.
- **Sampling.** Claude 5 models reject non-default temperature, so solver diversity there comes only from
  independent calls.

## Layout

```
src/satgen/
  pipeline.py   stages, gate, salvage, review modes, concurrency, checkpointing, stats
  evaluate.py   three-way review scoring, thresholds, anchoring
  answers.py    strict parsing of letters, numbers and booleans
  dedupe.py     MinHash + LSH index
  llm.py        Gemini and Claude clients, retry with backoff, simulated models
  cost.py       thread-safe token and cost accounting
  cli.py        `run` and `eval` commands
scripts/build_eval_set.py   builds the labelled set
```
