# satgen: multi-model Digital SAT question pipeline

[![ci](https://github.com/Dulkinsar/sat-question-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Dulkinsar/sat-question-pipeline/actions/workflows/ci.yml)

Generates Digital SAT questions with a cheap model, filters them by having a solver answer each one
several times, sends the survivors to a stronger model for review, confirms any answer-key change with a
blind re-solve, deduplicates against the existing bank, and exports JSON for bulk upload.

> **About this repo.** I built the original pipeline in 2025 at nFactorial Academy, where it produced
> 1,800 questions for a live question bank. That code belongs to the company. This is a public rebuild
> written on 13-14 October 2026 with Claude Code (AI-assisted coding), extended with an evaluation harness
> and the robustness work the original lacked. The history is squashed into one commit.

```
generate     Gemini 2.5 Flash-Lite, JSON mode, curated style context
  -> shuffle    choice order randomised per question; the key moves with its text
  -> solve x5   solver answers without seeing the key
  -> gate       >= 3/5 votes match the key ............................ -> review
                key fails, but >= 3/5 agree on another answer (salvage) -> review as suspected key error
                otherwise ............................................... screened out
  -> review     Claude Sonnet: accept, reject, or correct the key (with a new explanation)
  -> re-check   a correction ships only if the review model, blind to key and votes, agrees 2 of 3 times
  -> dedupe     MinHash + LSH against this run and the existing bank
  -> export     accepted questions as JSON; disputed items listed for a human
```

## Background

After the Digital SAT launched, practice material in the new format stayed scarce for a long time. The
platform at nFactorial served around 70 students on a typical day, and they worked through new questions
faster than a team could write them. The budget was small, so each question had to be cheap and still
good enough for students who were already studying on the platform daily.

The first batches came from a single model and were poor: wrong answer keys, ambiguous wording and weak
distractors, all reaching real students. Students could report a question as incorrect, and the share of
unique questions reported in a week was the quality metric. With nearly every question seen within a
week, that share started at roughly 20% for verbal and just under half for math. After the two-model
pipeline it stayed under 10%.

### The original and the rebuild

The 2025 original used the same roles: Gemini 2.5 Flash-Lite generated each question and solved it five
times, questions with at least 3 of 5 matching votes went to Claude 3.7 Sonnet for review, and the weekly
share of student-reported questions was the quality signal. This rebuild keeps that core and adds:

- a reviewer on Claude Sonnet 5.5, since 3.7 Sonnet was retired in February 2026
- salvage of suspected key errors and a blind re-check of every key correction
- a labelled evaluation set with planted errors, scored per threshold and per configuration
- concurrent workers, retries with backoff, and a checkpoint that resumes after a crash
- MinHash/LSH deduplication against the run and the existing bank
- student-produced response (numeric) math alongside multiple choice

## Design decisions

- **Gate before review.** Five Flash-Lite solves cost less than one Sonnet review, so only questions that
  survive the gate reach the reviewer. The report splits spend by stage.
- **Why a gate.** A question a model can't answer consistently is usually ambiguous or mis-keyed. The
  threshold is a parameter and `satgen eval` shows what each value buys.
- **Salvage.** If the key fails the gate but the solvers agree on another answer, the key is the likely
  fault. Those questions go to review as suspected key errors instead of being thrown away.
- **Blind re-check.** The reviewer sees the solver votes and can anchor on them. A key change ships only
  if the review model, solving with no key and no votes, agrees in 2 of 3 attempts; otherwise a human decides.
- **Correlated errors.** By default the generator also solves, as in the original, so one model's
  misconception can pass the gate. `--solver-model` swaps in a different model.
- **Shuffling.** Generators put the key in slot A far too often, so choices are shuffled per question.
- **Parsing.** Solvers answer in JSON. If a model ignores that, only the first line is read, uppercase
  letters only, and anything unclear counts as `?`. Numeric answers handle Unicode minus, LaTeX fractions
  and rounded decimals (0.333 matches 1/3).
- **Statuses.** Every attempt ends as `generation_failed`, `error`, `screened_out`, `rejected`,
  `needs_human`, `duplicate` or `accepted`, so rates are computed over everything attempted.

## Evaluation

`data/eval/eval_set.jsonl` holds 234 labelled items built by `scripts/build_eval_set.py`:

| Label | Count | Meaning |
|---|---|---|
| `good` | 83 | correct key; should ship |
| `bad_key` | 83 | the same items with the key moved to a distractor; must not ship with that key |
| `ambiguous` | 68 | one distractor replaced by a second defensible answer; must not ship |

Verbal items (words in context, transitions) are hand-written. Math items come from templates, so their
answers are correct by construction and their distractors model real student slips (sign errors,
forgetting to divide). Each item is solved five times and reviewed once, so every threshold and
configuration is scored from the same calls. The reviewer always sees the solver votes, as in
production, so the `review_only` row measures its decisions with votes in view:

```bash
python -m satgen eval                              # solver = generator model
python -m satgen eval --solver-model claude-haiku-5-5   # independent solver
```

### Results

Run on 14 October 2026 with Claude Haiku 5.5 as the solver (5 runs per item) and Claude Sonnet 5.5 as the
reviewer. Raw per-item output is in [`eval/results/`](eval/results/). Threshold k = 3 of 5:

| Configuration | Shipped | Error rate of shipped | Good items kept | Bad keys caught | Ambiguous caught | Cost (234 items) |
|---|---|---|---|---|---|---|
| Gate only | 138 | 0.406 | 0.988 | 0.988 | 0.191 | $0.06 |
| Review only | 86 | 0.0 | 1.0 | 1.0 | 1.0 | $1.31 |
| Gate + review | 82 | 0.0 | 0.988 | 1.0 | 1.0 | $0.80 |
| Full (with blind re-check) | 85 | 0.0 | 0.988 | 1.0 | 1.0 | $1.31 |

What the numbers show:

- The solver gate catches wrong keys (0.988) and misses ambiguous questions (0.191 caught), because five
  solvers can agree on one reading of a question that has two defensible answers. That gap is the
  reason the reviewer exists.
- Gate + review costs 39% less than review alone ($0.80 against $1.31), because the gate sends the
  reviewer only the questions that survive it. The review-only cost includes the solver calls, since the
  reviewer is shown their votes.
- The full design adds salvage and the blind re-check. It ships 3 more questions than gate + review (85
  against 82) and costs $1.31, the same as review alone on this set. On this set the gate saves money only
  when salvage and the re-check are left out; what the full design buys is recovered questions and
  protection against bad corrections (85 corrections were proposed and 83 were right).

Read these with the set's limits in mind. The planted errors are blunt, the math items come from
templates, and the reviewer (a Claude model) judges questions that a Claude model solved, so a perfect
score on this set shows the harness separates the cases cleanly and says little about subtle real-world
errors. A real run on freshly generated questions, with a human labelling a sample, is the next
measurement.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                       # no network needed
python -m satgen run --offline --count 36    # end-to-end run with simulated models
python -m satgen eval --simulate             # harness smoke test (scores are meaningless)
```

The simulated models plant wrong keys in about 15% of questions, so the offline run exercises the gate,
salvage, corrections and dedupe. Its numbers show the mechanics only.

With real models:

```bash
pip install -e ".[live]"
cp .env.example .env && set -a && source .env && set +a
python -m satgen run --count 200 --bank my_existing_bank.json
```

A run writes each finished question to `out/checkpoint.jsonl`. Rerunning the same command resumes, and
questions that failed on API errors are retried. The checkpoint records the models and settings that
wrote it and refuses to resume under different ones; a line cut short by a crash is skipped and
repaired. `--fresh` starts over.

## Outputs

`out/questions.json` holds the accepted questions and `out/report.json` the stats: status counts, cost per
model and stage, items that need a human, and reworded pairs. `examples/` has a sample offline run.

## Layout

```
src/satgen/
  pipeline.py   stages, gate and salvage logic, concurrency, checkpointing, stats
  answers.py    strict parsing of letters, numbers and booleans
  dedupe.py     MinHash + LSH index
  evaluate.py   threshold and ablation scoring on the labelled set
  llm.py        Gemini and Claude clients, retry with backoff, simulated models
  cost.py       thread-safe token and cost accounting
  cli.py        `run` and `eval` commands
scripts/build_eval_set.py   builds the labelled set
tests/                      gate, re-check, resume, dedupe, parsing
```

## Limitations

- The evaluation set is small (234 items) and the verbal half is written by one person, so its labels
  carry that person's judgement.
- Question types that depend on figures, tables or graphs are out of scope.
- Dedupe compares wording. Two questions that test the same idea with different text both pass.
