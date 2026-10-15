# The original pipeline at nFactorial

After the Digital SAT launched, practice material in the new format stayed scarce for a long time. The
platform at nFactorial Academy served around 70 students on a typical day, and they worked through new
questions faster than a team could write them. The budget was small, so each question had to be cheap
and still good enough for students who were studying on the platform daily.

The first batches came from a single model and were poor: wrong answer keys, ambiguous wording and weak
distractors, all reaching real students. Students could report a question as incorrect, and the share of
unique questions reported in a week was the quality metric. With nearly every question seen within a
week, that share started at roughly 20% for verbal and just under half for math.

The 2025 pipeline used the roles this repo keeps: Gemini 2.5 Flash-Lite generated each question and
solved it five times, and questions with at least 3 of 5 matching votes went to Claude 3.7 Sonnet for
review. After it went live the weekly report share stayed under 10%, and the pipeline produced 1,800
questions for the live bank.

## What the rebuild adds

- a reviewer on Claude Sonnet 5.5 (3.7 Sonnet was retired in February 2026), with a split verdict and a
  key-hidden mode
- salvage of suspected key errors and a blind re-check of every key correction
- a labelled evaluation set with planted errors, scored per threshold, per configuration and per review mode
- concurrent workers, retries with backoff, and a checkpoint that resumes after a crash
- MinHash/LSH deduplication against the run and the existing bank
- student-produced response (numeric) math alongside multiple choice

Out of scope: question types that depend on figures, tables or graphs, and deduplication by idea rather
than wording.
