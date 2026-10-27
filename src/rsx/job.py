"""One extraction job: prepare the run, start (or connect to) vLLM, run the async client.

Resume-safe: rerunning the same command only processes what is not done yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .client import RunContext, make_client, run_extraction
from .data import WorkUnit, build_work_units, longest_first, select_units, shard_units
from .feeders import ReportFeeder, make_feeder, parse_feeder_options
from .paths import Paths
from .registry import ModelSpec, RequestDefaults, get_model
from .results import (
    ResultsWriter,
    RunLock,
    latest_records,
    pending_units,
    results_filename,
)
from .server import ServerDiedError, ServerTimeoutError, VllmServer
from .stats import ProgressTracker, summarize
from .template import Template, load_template

log = logging.getLogger(__name__)

EXIT_OK, EXIT_REMAINING, EXIT_BREAKER, EXIT_SERVER = 0, 3, 4, 5
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S"
    )
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@dataclass
class Prepared:
    template: Template
    spec: ModelSpec
    feeder: ReportFeeder
    units: list[WorkUnit]  # after shard + selection
    n_total: int  # all units of the input
    run_dir: Path
    results_path: Path
    params: RequestDefaults
    num_shards: int
    shard_id: int


def _shards(args) -> tuple[int, int]:
    if args.num_shards:
        return args.num_shards, args.shard_id or 0
    if os.environ.get("SLURM_ARRAY_TASK_ID") and os.environ.get("SLURM_ARRAY_TASK_COUNT"):
        task = int(os.environ["SLURM_ARRAY_TASK_ID"]) - int(
            os.environ.get("SLURM_ARRAY_TASK_MIN", "0")
        )
        return int(os.environ["SLURM_ARRAY_TASK_COUNT"]), task
    return 1, 0


def request_overrides(args) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "timeout_s": args.request_timeout,
        "seed": args.seed_request,
    }


def prepare(args, paths: Paths) -> Prepared:
    template = load_template(args.template, paths.templates_dir)
    for w in template.warnings:
        log.warning("template: %s", w)
    spec = get_model(args.model)
    feeder_name = args.feeder or template.config.feeder
    if not feeder_name:
        raise ValueError(
            "no feeder: pass --feeder (e.g. ct_rate) or set `feeder:` in the template config.yaml"
        )
    options = {**template.config.feeder_options, **parse_feeder_options(args.feeder_opt)}
    feeder = make_feeder(feeder_name, Path(args.input).resolve(), options)
    all_units = build_work_units(feeder.iter_units(), text_dedup=not args.no_text_dedup)
    num_shards, shard_id = _shards(args)
    units = shard_units(all_units, num_shards, shard_id)
    units = select_units(units, limit=args.limit, sample=args.sample, seed=args.seed, ids=args.ids)
    run_name = args.run_name or Path(args.input).stem
    run_dir = paths.runs_dir / template.name / spec.name / run_name
    params = spec.request.merged(template.config.request).merged(request_overrides(args))
    return Prepared(
        template,
        spec,
        feeder,
        units,
        len(all_units),
        run_dir,
        run_dir / results_filename(num_shards, shard_id),
        params,
        num_shards,
        shard_id,
    )


def _gpu_info() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def _update_run_yaml(prep: Prepared, args, job_id: str) -> None:
    path = prep.run_dir / "run.yaml"
    info = yaml.safe_load(path.read_text()) if path.exists() else {}
    info.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    info.update(
        {
            "template": prep.template.name,
            "template_source": str(prep.template.folder or prep.template.files["template"]),
            "template_hash": prep.template.hash,
            "model": prep.spec.name,
            "input": str(Path(args.input).resolve()),
            "feeder": prep.feeder.describe(),
            "member_column": prep.feeder.member_column,
            "request": prep.params.model_dump(),
            "units_in_input": prep.n_total,
        }
    )
    jobs = info.setdefault("jobs", [])
    jobs.append(
        {
            "job_id": job_id,
            "started": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "shard": f"{prep.shard_id}/{prep.num_shards}",
            "selected_units": len(prep.units),
        }
    )
    path.write_text(yaml.safe_dump(info, sort_keys=False, allow_unicode=True))


class _StopFlag:
    """Set by SIGTERM/SIGINT/SIGUSR1 (Slurm --signal, scancel, preemption, Ctrl-C)."""

    def __init__(self):
        self.set_ = False
        self.event: asyncio.Event | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    def __call__(self) -> bool:
        return self.set_

    def trigger(self, signum, _frame=None) -> None:
        if not self.set_:
            log.warning(
                "received signal %s: finishing in-flight requests, then stopping",
                signal.Signals(signum).name,
            )
        self.set_ = True
        if self.event is not None and self.loop is not None:
            self.loop.call_soon_threadsafe(self.event.set)

    def install(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(sig, self.trigger)


def run_job(args, paths: Paths, *, connect: dict[str, Any] | None = None) -> int:
    """Full job. connect=None: start a vLLM server here;
    else connect = {"base_url", "api_key", "served_name"} of a running server."""
    prep = prepare(args, paths)
    job_id = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("local-%Y%m%d-%H%M%S")
    if prep.num_shards > 1:
        job_id = f"{job_id}_s{prep.shard_id}"
    prep.run_dir.mkdir(parents=True, exist_ok=True)
    lock_name = ".lock" if prep.num_shards <= 1 else f".lock.s{prep.shard_id}of{prep.num_shards}"
    stop = _StopFlag()
    stop.install()

    with RunLock(prep.run_dir / lock_name):
        records = latest_records(prep.run_dir)
        pending = pending_units(
            prep.units,
            records,
            template_hash=prep.template.hash,
            retry_failed=not args.no_retry_failed,
            allow_template_change=args.allow_template_change,
        )
        snap = prep.run_dir / "template"
        if snap.exists():
            shutil.rmtree(snap)
        prep.template.snapshot(snap)
        _update_run_yaml(prep, args, job_id)
        log.info("run dir %s", prep.run_dir)
        log.info(
            "input %s: %d units (%d selected for shard %d/%d), %d already done, %d pending",
            args.input,
            prep.n_total,
            len(prep.units),
            prep.shard_id,
            prep.num_shards,
            len(prep.units) - len(pending),
            len(pending),
        )
        if not pending:
            log.info("nothing to do")
            if args.then_export:
                _export(prep.run_dir)
            return EXIT_OK

        job_dir = prep.run_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(job_dir / "client.log")
        fh.setFormatter(logging.Formatter(LOG_FORMAT, "%H:%M:%S"))
        logging.getLogger().addHandler(fh)

        server: VllmServer | None = None
        env_info: dict[str, Any] = {
            "job_id": job_id,
            "host": socket.gethostname(),
            "gpus": _gpu_info(),
            "sif": str(paths.sif),
            "model": prep.spec.name,
            "args": vars(args),
        }
        t_job = time.monotonic()
        try:
            if connect is None:
                server = VllmServer(paths, prep.spec, job_id=os.environ.get("SLURM_JOB_ID"))
                server.start(job_dir / "vllm.log")
                try:
                    server.wait_ready(stop=stop)
                except (ServerDiedError, ServerTimeoutError) as e:
                    log.error("vLLM server failed: %s", e)
                    return EXIT_REMAINING if stop() else EXIT_SERVER
                info = server.info()
                connect = {
                    "base_url": server.base_url,
                    "api_key": server.api_key,
                    "served_name": prep.spec.served_name,
                    "vllm_version": info.get("vllm_version"),
                }
                env_info["server"] = info
            env_info["connect"] = {k: v for k, v in connect.items() if k != "api_key"}
            (job_dir / "environment.json").write_text(json.dumps(env_info, indent=2, default=str))
            log.info(
                "vLLM %s at %s, %d pending units, concurrency %d",
                connect.get("vllm_version"),
                connect["base_url"],
                len(pending),
                args.concurrency or prep.spec.concurrency,
            )
            outcome, progress = asyncio.run(
                _run_async(
                    prep,
                    pending,
                    connect,
                    args,
                    job_id,
                    stop,
                    args.concurrency or prep.spec.concurrency,
                )
            )
        finally:
            if server is not None:
                server.stop()
            logging.getLogger().removeHandler(fh)
            fh.close()

        remaining = pending_units(
            prep.units,
            latest_records(prep.run_dir),
            template_hash=prep.template.hash,
            retry_failed=False,
            allow_template_change=True,
        )
        stats = {
            "job_id": job_id,
            "wall_s": round(time.monotonic() - t_job, 1),
            "server_startup_s": server.startup_s if server else None,
            "progress": progress.summary(),
            "outcome": outcome.__dict__,
            "remaining_units": len(remaining),
            "run_summary": summarize(latest_records(prep.run_dir).values()),
        }
        (job_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
        log.info(
            "job done: %s | remaining %d | %s",
            outcome.by_status,
            len(remaining),
            progress.summary(),
        )

    if args.then_export:
        _export(prep.run_dir)
    if outcome.breaker or outcome.fatal:
        return EXIT_BREAKER
    return EXIT_REMAINING if remaining else EXIT_OK


async def _run_async(
    prep: Prepared,
    pending: list[WorkUnit],
    connect: dict[str, Any],
    args,
    job_id: str,
    stop: _StopFlag,
    concurrency: int,
):
    loop = asyncio.get_running_loop()
    event = asyncio.Event()
    stop.event, stop.loop = event, loop
    if stop():
        event.set()
    client = make_client(
        connect["base_url"], connect["api_key"], concurrency, prep.params.timeout_s
    )
    ctx = RunContext(
        client=client,
        template=prep.template,
        served_name=connect["served_name"],
        model_name=prep.spec.name,
        params=prep.params,
        vllm_version=connect.get("vllm_version"),
        job_id=job_id,
        keep_reasoning=args.keep_reasoning,
    )
    progress = ProgressTracker(len(pending))
    try:
        with ResultsWriter(prep.results_path) as writer:
            outcome = await run_extraction(
                ctx,
                longest_first(pending),
                writer,
                concurrency=concurrency,
                stop=event,
                on_record=progress.update,
                drain_timeout_s=args.drain_timeout,
            )
    finally:
        await client.close()
    progress.log()
    return outcome, progress


def _export(run_dir: Path) -> None:
    from .export import export_run

    try:
        written = export_run(run_dir)
        log.info("exports in %s (%d files)", run_dir / "exports", len(written))
    except Exception:
        log.exception(
            "export failed (results are safe in results*.jsonl; rerun `rsx export %s`)", run_dir
        )


def serve(args, paths: Paths) -> int:
    """Start a server and keep it up until a signal (interactive use: salloc + rsx serve)."""
    spec = get_model(args.model)
    job_id = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("local-%Y%m%d-%H%M%S")
    stop = _StopFlag()
    stop.install()
    server = VllmServer(paths, spec, job_id=os.environ.get("SLURM_JOB_ID"), port=args.port)
    servers_dir = paths.runs_dir / "servers"
    servers_dir.mkdir(parents=True, exist_ok=True)
    conn_file = servers_dir / f"{job_id}.json"
    try:
        server.start(servers_dir / f"{job_id}.vllm.log")
        server.wait_ready(stop=stop)
        info = server.info()
        conn = {**server.connection_info(), "vllm_version": info.get("vllm_version")}
        conn_file.write_text(json.dumps(conn, indent=2))
        conn_file.chmod(0o600)
        log.info(
            "server ready: %s (connection file %s). Ctrl-C / scancel to stop.",
            server.base_url,
            conn_file,
        )
        log.info("client: rsx run --server %s --template ... --input ...", conn_file)
        while not stop() and server.proc is not None and server.proc.poll() is None:
            time.sleep(2)
        if server.proc is not None and server.proc.poll() is not None:
            log.error(
                "vLLM exited (code %s). Log tail:\n%s", server.proc.returncode, server.log_tail()
            )
            return EXIT_SERVER
        return EXIT_OK
    except (ServerDiedError, ServerTimeoutError) as e:
        log.error("vLLM server failed: %s", e)
        return EXIT_SERVER
    finally:
        server.stop()
        conn_file.unlink(missing_ok=True)


def dry_run(args, paths: Paths) -> int:
    """No server: show what would be sent for the first selected unit."""
    prep = prepare(args, paths)
    records = latest_records(prep.run_dir) if prep.run_dir.exists() else {}
    pending = pending_units(
        prep.units,
        records,
        template_hash=prep.template.hash,
        allow_template_change=True,
        retry_failed=not args.no_retry_failed,
    )
    print(f"template {prep.template.name} (hash {prep.template.hash}), model {prep.spec.name}")
    print(
        f"input {args.input}: {prep.n_total} units, {len(prep.units)} selected, "
        f"{len(pending)} pending"
    )
    print(f"run dir {prep.run_dir}")
    print(f"request params {prep.params.model_dump()}")
    if not prep.units:
        return EXIT_OK
    unit = longest_first(prep.units)[0]
    messages = prep.template.render_messages(unit.text)
    schema = json.dumps(prep.template.json_schema)
    chars = sum(len(m["content"]) for m in messages)
    print(f"\nlongest unit {unit.id} members={list(unit.members)[:4]} keys={list(unit.keys)[:4]}")
    print(f"prompt ~{chars} chars (~{chars // 4} tokens), schema {len(schema)} chars")
    for m in messages:
        print(f"\n----- {m['role']} -----\n{m['content']}")
    return EXIT_OK
