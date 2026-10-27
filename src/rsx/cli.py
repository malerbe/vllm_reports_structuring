"""rsx command line. `rsx <command> --help` for details."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from .paths import load_paths


def _add_job_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("what to structure")
    g.add_argument(
        "--template",
        "-t",
        required=True,
        help="template name (templates/<name>), folder or .yaml file",
    )
    g.add_argument(
        "--model", "-m", default="gpt-oss-120b", help="model name from configs/models.yaml"
    )
    g.add_argument("--input", "-i", required=True, help="input file (e.g. a CT-RATE reports CSV)")
    g.add_argument(
        "--feeder", "-f", help="dataset reader (default: the template's config.yaml feeder)"
    )
    g.add_argument(
        "--feeder-opt", action="append", metavar="KEY=VALUE", help="feeder option (repeatable)"
    )
    g.add_argument(
        "--no-text-dedup", action="store_true", help="do not merge units with identical text"
    )
    g = p.add_argument_group("run and subset")
    g.add_argument(
        "--run-name", help="output folder under runs/<template>/<model>/ (default: input file stem)"
    )
    g.add_argument("--limit", type=int, help="first N units only")
    g.add_argument(
        "--sample", type=int, help="random sample of N units (deterministic with --seed)"
    )
    g.add_argument("--seed", type=int, default=0, help="seed of --sample")
    g.add_argument("--ids", nargs="+", help="only these unit ids / feeder keys / member ids")
    g.add_argument(
        "--num-shards", type=int, help="split units in N shards (default: Slurm array size)"
    )
    g.add_argument(
        "--shard-id", type=int, help="shard processed by this job (default: Slurm array index)"
    )
    g.add_argument(
        "--no-retry-failed", action="store_true", help="do not redo units recorded as failed"
    )
    g.add_argument(
        "--allow-template-change",
        action="store_true",
        help="resume a run although the template changed (units made with the old one are redone)",
    )
    g = p.add_argument_group("client")
    g.add_argument("--concurrency", type=int, help="requests in flight (default: models.yaml)")
    g.add_argument("--temperature", type=float)
    g.add_argument("--max-tokens", type=int)
    g.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    g.add_argument("--request-timeout", type=float, help="seconds per request")
    g.add_argument("--seed-request", type=int, help="sampling seed sent with each request")
    g.add_argument(
        "--drain-timeout",
        type=float,
        default=120.0,
        help="seconds to finish in-flight requests on stop",
    )
    g.add_argument(
        "--keep-reasoning",
        action="store_true",
        help="store the model's reasoning text in the records",
    )
    g.add_argument("--then-export", action="store_true", help="export tables at the end")


def _cmd_job(args) -> int:
    from .job import run_job

    return run_job(args, load_paths())


def _cmd_run(args) -> int:
    from .job import dry_run, run_job

    paths = load_paths()
    if args.dry_run:
        return dry_run(args, paths)
    if args.server:
        conn = json.loads(Path(args.server).read_text())
        if conn.get("model") and conn["model"] != args.model:
            print(f"note: server file is for model {conn['model']}, using it", file=sys.stderr)
            args.model = conn["model"]
    elif args.base_url:
        from .registry import get_model

        conn = {
            "base_url": args.base_url,
            "api_key": args.api_key or "EMPTY",
            "served_name": args.served_name or get_model(args.model).served_name,
        }
    else:
        print(
            "rsx run needs --server FILE, --base-url URL or --dry-run "
            "(use `rsx job` to start a server)",
            file=sys.stderr,
        )
        return 2
    return run_job(args, paths, connect=conn)


def _cmd_serve(args) -> int:
    from .job import serve

    return serve(args, load_paths())


def _cmd_submit(args) -> int:
    paths = load_paths()
    sbatch = [
        "sbatch",
        "--partition",
        args.partition,
        "--time",
        args.time,
        "--job-name",
        args.job_name,
    ]
    if args.array:
        sbatch += ["--array", f"0-{args.array - 1}"]
    if args.dependency:
        sbatch += ["--dependency", args.dependency]
    job_args = [
        "job",
        "--template",
        args.template,
        "--model",
        args.model,
        "--input",
        str(Path(args.input).resolve()),
        *args.extra,
    ]
    cmd = [*sbatch, str(paths.root / "scripts" / "run.sbatch"), *job_args]
    print("$ " + shlex.join(cmd))
    if args.print_only:
        return 0
    out = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    if out.returncode == 0:
        job_id = out.stdout.strip().split()[-1]
        print(f"log: {paths.runs_dir / 'logs'}/{args.job_name}_{job_id}.log")
    return out.returncode


def _cmd_template(args) -> int:
    from .template import field_catalog, load_template

    paths = load_paths()
    t = load_template(args.template, paths.templates_dir)
    if args.action == "schema":
        print(json.dumps(t.json_schema, indent=2))
    elif args.action == "show":
        print(t.render_messages("{{report}}")[0]["content"])
        print("----- user -----")
        print(t.user_prompt)
    elif args.action == "fields":
        for f in field_catalog(t.root):
            vals = f" {f['values']}" if f["values"] else ""
            print(f"{f['path']:<45} {f['kind']:<8}{' null' if f['nullable'] else '     '}{vals}")
    else:  # check
        catalog = field_catalog(t.root)
        schema = json.dumps(t.json_schema)
        print(f"template      {t.name}  (hash {t.hash})")
        print(f"files         {', '.join(f'{k}={v}' for k, v in t.files.items())}")
        print(
            f"fields        {len(catalog)} leaves "
            f"({sum(f['kind'] != 'text' for f in catalog)} typed)"
        )
        print(f"system prompt {len(t.system_prompt)} chars (~{len(t.system_prompt) // 4} tokens)")
        print(f"json schema   {len(schema)} chars, portable keywords only")
        print(f"feeder        {t.config.feeder or '(none: pass --feeder)'}")
        print(f"export tables {t.config.export_tables or '-'}")
        for w in t.warnings:
            print(f"warning       {w}")
        print("OK")
    return 0


def _cmd_export(args) -> int:
    from .export import export_run

    written = export_run(Path(args.run_dir), explode=not args.no_explode)
    for name, path in written.items():
        print(f"{name:<22} {path}")
    return 0


def _cmd_stats(args) -> int:
    from .results import latest_records
    from .stats import summarize

    print(json.dumps(summarize(latest_records(Path(args.run_dir)).values()), indent=2))
    return 0


def _cmd_models(args) -> int:
    from .download import check_model_dir
    from .registry import load_registry

    paths = load_paths()
    for name, spec in load_registry().items():
        path = spec.model_path(paths.models_dir)
        state = "ok" if path.is_dir() and not check_model_dir(path) else "missing"
        print(f"{name:<16} {spec.hf_repo:<24} {state:<8} {path}")
    print(f"vLLM image: {paths.sif} ({'ok' if paths.sif.is_file() else 'missing'})")
    return 0


def _cmd_feeders(args) -> int:
    from .feeders import available_feeders

    for name, cls in available_feeders().items():
        doc = (cls.__doc__ or "").strip().splitlines()[0]
        print(f"{name:<10} {cls.__name__:<20} {doc}")
        for k, v in cls.default_options.items():
            print(f"{'':<10}   {k} = {v!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsx", description="Structure radiology reports with vLLM."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("job", help="start vLLM + run the extraction (inside a Slurm GPU job)")
    _add_job_args(p)
    p.set_defaults(func=_cmd_job)

    p = sub.add_parser("run", help="client only, against a running server (rsx serve) or --dry-run")
    _add_job_args(p)
    p.add_argument(
        "--server", help="connection file written by `rsx serve` (runs/servers/<job>.json)"
    )
    p.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--api-key")
    p.add_argument("--served-name")
    p.add_argument(
        "--dry-run", action="store_true", help="no server: print the prompt of the longest unit"
    )
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("serve", help="start a vLLM server and keep it up (interactive allocation)")
    p.add_argument("--model", "-m", default="gpt-oss-120b")
    p.add_argument("--port", type=int)
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("submit", help="submit `rsx job` with sbatch (login node)")
    p.add_argument("--template", "-t", required=True)
    p.add_argument("--model", "-m", default="gpt-oss-120b")
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--partition", "-p", default="h200", help="h200 or preempt")
    p.add_argument("--time", default="04:00:00")
    p.add_argument("--job-name", default="rsx")
    p.add_argument("--array", type=int, help="N array tasks = N shards")
    p.add_argument("--dependency", help="e.g. afterany:<jobid> to chain jobs")
    p.add_argument("--print-only", action="store_true")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="after --: more `rsx job` options")
    p.set_defaults(func=_cmd_submit)

    p = sub.add_parser("template", help="inspect a template")
    p.add_argument("action", choices=["check", "show", "schema", "fields"])
    p.add_argument("template")
    p.set_defaults(func=_cmd_template)

    p = sub.add_parser("export", help="write tables (parquet/csv) for a run directory")
    p.add_argument("run_dir")
    p.add_argument(
        "--no-explode", action="store_true", help="one row per unit instead of per member"
    )
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("stats", help="summary of a run directory")
    p.add_argument("run_dir")
    p.set_defaults(func=_cmd_stats)

    sub.add_parser("models", help="registered models and download state").set_defaults(
        func=_cmd_models
    )
    sub.add_parser("feeders", help="available dataset readers and their options").set_defaults(
        func=_cmd_feeders
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from .job import setup_logging

    args = build_parser().parse_args(argv)
    if getattr(args, "extra", None) and args.extra[0] == "--":
        args.extra = args.extra[1:]
    setup_logging(args.verbose)
    return args.func(args)
