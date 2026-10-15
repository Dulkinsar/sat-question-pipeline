"""Command line entry: `python -m satgen run` generates a batch, `python -m satgen eval`
scores the gate and reviewer.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import evaluate, pipeline
from .llm import GEN_MODEL, REVIEW_MODEL, SimulatedClient, make_client

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TYPES = [
    "words_in_context",
    "text_structure",
    "central_ideas",
    "transitions",
    "boundaries",
    "linear_equations",
    "spr_linear_equations",
]
OFFLINE_TYPES = ["words_in_context", "words_in_context", "spr_linear_equations"]


def _resolve(path: Path) -> Path:
    """Look relative to the working directory first, then to the repo root."""
    return path if path.exists() or path.is_absolute() else ROOT / path


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--runs", type=int, default=5, help="solver runs per question")
    p.add_argument("--threshold", type=int, default=3, help="agreeing runs needed to reach review")
    p.add_argument("--concurrency", type=int, default=8, help="questions processed in parallel")
    p.add_argument("--solver-model", default=None, help="defaults to the generator model")
    p.add_argument("--review-model", default=REVIEW_MODEL)
    p.add_argument("-v", "--verbose", action="store_true")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"run", "eval", "-h", "--help"}:
        argv.insert(0, "run")

    parser = argparse.ArgumentParser(prog="satgen", description="Digital SAT question pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="generate, verify and export a batch")
    _common(r)
    r.add_argument("--count", type=int, default=20, help="questions to attempt")
    r.add_argument("--types", nargs="*", default=None)
    r.add_argument("--gen-model", default=GEN_MODEL)
    r.add_argument("--context-file", type=Path, default=Path("data/style_context.txt"))
    r.add_argument("--bank", type=Path, help="existing questions (export format) to dedupe against")
    r.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="JSONL of finished questions; rerunning resumes from it "
        "(default out/checkpoint.jsonl, or out/checkpoint.offline.jsonl with --offline)",
    )
    r.add_argument("--fresh", action="store_true", help="ignore and replace an existing checkpoint")
    r.add_argument("--no-salvage", action="store_true", help="discard every question that fails the gate")
    r.add_argument(
        "--review-mode",
        choices=["keyless", "key"],
        default="keyless",
        help="key: reviewer sees the key and solver votes; keyless: it sees neither and its answer is compared in code",
    )
    r.add_argument("--seed", type=int, default=0, help="seed for choice shuffling")
    r.add_argument("--out", type=Path, default=Path("out/questions.json"))
    r.add_argument("--report", type=Path, default=Path("out/report.json"))
    r.add_argument("--offline", action="store_true", help="simulated models, no API keys")

    e = sub.add_parser("eval", help="score gate thresholds and reviewer on the labelled set")
    _common(e)
    e.add_argument("--items", type=Path, default=Path("data/eval/eval_set.jsonl"))
    e.add_argument("--limit", type=int, default=None, help="evaluate only the first N items")
    e.add_argument("--rescore", type=Path, help="score a saved .items.jsonl again; makes no API calls")
    e.add_argument("--out", type=Path, default=Path("eval/results"))
    e.add_argument(
        "--simulate",
        action="store_true",
        help="smoke test with an oracle that reads the labels (scores are meaningless)",
    )

    args = parser.parse_args(argv)
    if not 1 <= args.threshold <= args.runs:
        parser.error(f"--threshold must be between 1 and --runs ({args.runs})")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return _run(args) if args.cmd == "run" else _eval(args)


def _run(args) -> int:
    ctx_path = _resolve(args.context_file)
    context = ctx_path.read_text(encoding="utf-8") if ctx_path.exists() else ""
    if not context:
        logging.warning("no style context found at %s", ctx_path)
    if args.offline:
        gen = solver = SimulatedClient("simulated-generator")
        reviewer = SimulatedClient("simulated-reviewer")
        types = args.types or OFFLINE_TYPES
    else:
        gen = make_client(args.gen_model)
        solver = make_client(args.solver_model) if args.solver_model else gen
        reviewer = make_client(args.review_model)
        types = args.types or DEFAULT_TYPES

    if args.checkpoint is None:
        args.checkpoint = Path("out/checkpoint.offline.jsonl" if args.offline else "out/checkpoint.jsonl")
    if args.fresh and args.checkpoint.exists():
        args.checkpoint.unlink()
    config = pipeline.Config(
        runs=args.runs,
        threshold=args.threshold,
        concurrency=args.concurrency,
        seed=args.seed,
        salvage=not args.no_salvage,
        review_mode=args.review_mode,
    )
    bank = pipeline.load_bank(args.bank) if args.bank else []
    p = pipeline.Pipeline(gen, solver, reviewer, context=context, config=config)
    try:
        result = p.run(types, args.count, checkpoint=args.checkpoint, bank=bank)
    except pipeline.CheckpointMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    n = pipeline.export(result.records, args.out)
    by_index = {r.index: r for r in result.records}
    report = {
        "models": {"generator": gen.model, "solver": solver.model, "reviewer": reviewer.model},
        "config": config.__dict__,
        "stats": pipeline.stats(result.records, runs=args.runs),
        # Spend covers this session only, so divide by questions accepted in this session.
        "cost": result.tracker.summary(
            accepted=sum(r.status == "accepted" and r.index in result.processed for r in result.records)
        ),
        "manual_review": {
            "needs_human": [
                {
                    "index": r.index,
                    "stem": r.question.stem,
                    "key": r.original_answer,
                    "votes": r.solve_votes,
                    "reviewer": r.verdict.__dict__ if r.verdict else None,
                    "recheck": r.recheck_votes,
                    "why": r.error,
                }
                for r in result.records
                if r.status == "needs_human"
            ],
            "similar_pairs": [
                {
                    "index": rec.index,
                    "stem": rec.question.stem,
                    "similar_to": key,
                    "similar_stem": by_index[int(key.split(":")[1])].question.stem if key.startswith("run:") else None,
                    "jaccard": s,
                }
                for rec, key, s in result.flagged
            ],
        },
        "errors": [{"index": r.index, "status": r.status, "error": r.error} for r in result.records if r.error],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["stats"], indent=2))
    print(json.dumps(report["cost"], indent=2))
    print(f"exported {n} questions to {args.out}; report at {args.report}")
    return 0


def _rescore(args) -> int:
    results = evaluate.load_results(args.rescore)
    config = pipeline.Config(runs=args.runs, threshold=args.threshold, concurrency=args.concurrency)
    rep = evaluate.report(results, config)
    solver, _, reviewer = args.rescore.name.removesuffix(".items.jsonl").partition("__")
    rep["models"] = {"solver": solver, "reviewer": reviewer}
    rep["simulated"] = False
    stem = args.rescore.name.removesuffix(".items.jsonl")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{stem}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = evaluate.to_markdown(rep)
    (args.out / f"{stem}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _eval(args) -> int:
    if args.rescore:
        return _rescore(args)
    items = evaluate.load_items(_resolve(args.items))[: args.limit]
    if args.simulate:
        solver = reviewer = evaluate.OracleSimClient(items)
    else:
        solver = make_client(args.solver_model or GEN_MODEL)
        reviewer = make_client(args.review_model)
    config = pipeline.Config(runs=args.runs, threshold=args.threshold, concurrency=args.concurrency)
    results = evaluate.run_items(items, solver, reviewer, config)
    rep = evaluate.report(results, config)
    rep["models"] = {"solver": solver.model, "reviewer": reviewer.model}
    rep["simulated"] = bool(args.simulate)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = "simulated" if args.simulate else f"{solver.model}__{reviewer.model}"
    (args.out / f"{stem}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (args.out / f"{stem}.items.jsonl").write_text(
        "".join(json.dumps(r.__dict__) + "\n" for r in results), encoding="utf-8"
    )
    md = evaluate.to_markdown(rep)
    (args.out / f"{stem}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
