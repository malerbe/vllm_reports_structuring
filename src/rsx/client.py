"""Async extraction client: N requests in flight against the vLLM OpenAI-compatible server."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai
from pydantic import BaseModel, ValidationError

from .data import WorkUnit
from .registry import RequestDefaults
from .results import STATUS_OK, Record, ResultsWriter
from .template import Template

log = logging.getLogger(__name__)


class TransientExhaustedError(RuntimeError):
    """Server unreachable / overloaded after all retries: the unit is NOT recorded
    (it is redone by the next run)."""


class FatalClientError(RuntimeError):
    """Configuration problem (auth, unknown model...): stop the whole run."""


def make_client(
    base_url: str, api_key: str, concurrency: int, timeout_s: float
) -> openai.AsyncOpenAI:
    """trust_env=False: never route 127.0.0.1 through the login shell's http(s)_proxy.
    max_retries=0: our own retry policy is the only one."""
    http = httpx.AsyncClient(
        trust_env=False,
        limits=httpx.Limits(
            max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8
        ),
        timeout=httpx.Timeout(timeout_s, connect=30.0),
    )
    return openai.AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0, http_client=http)


def read_reasoning(message: Any) -> str:
    """vLLM >= 0.30 returns the gpt-oss analysis channel as `reasoning`
    (formerly `reasoning_content`)."""
    for name in ("reasoning", "reasoning_content"):
        value = getattr(message, name, None)
        if value is None:
            extra = getattr(message, "model_extra", None) or {}
            value = extra.get(name)
        if value:
            return str(value)
    return ""


def evaluate(
    content: str | None, finish_reason: str | None, model: type[BaseModel]
) -> tuple[str, Any, str | None]:
    """-> (status, parsed json or None, error message)."""
    if finish_reason == "length":
        return "truncated", None, "finish_reason=length (max_tokens reached)"
    if not content or not content.strip():
        return "empty", None, f"empty content (finish_reason={finish_reason})"
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        return "invalid_json", None, f"JSONDecodeError: {e}"
    try:
        model.model_validate(obj)
    except ValidationError as e:
        return "validation_error", None, f"{e.error_count()} validation error(s): {str(e)[:2000]}"
    return STATUS_OK, obj, None


@dataclass
class RunContext:
    """Everything a worker needs besides the unit."""

    client: openai.AsyncOpenAI
    template: Template
    served_name: str
    model_name: str
    params: RequestDefaults
    vllm_version: str | None = None
    job_id: str | None = None
    keep_reasoning: bool = False


async def call_once(ctx: RunContext, unit: WorkUnit, temperature: float, seed: int) -> Any:
    p = ctx.params
    kwargs: dict[str, Any] = {
        "model": ctx.served_name,
        "messages": ctx.template.render_messages(unit.text),
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "report", "schema": ctx.template.json_schema, "strict": True},
        },
        "max_completion_tokens": p.max_tokens,
        "temperature": temperature,
        "seed": seed,
        "timeout": p.timeout_s,
    }
    if p.top_p is not None:
        kwargs["top_p"] = p.top_p
    if p.reasoning_effort:
        kwargs["reasoning_effort"] = p.reasoning_effort
    if p.extra_body:
        kwargs["extra_body"] = p.extra_body
    return await ctx.client.chat.completions.create(**kwargs)


async def extract_one(ctx: RunContext, unit: WorkUnit) -> Record:
    """Structure one unit. Model failures are retried (new seed, higher temperature) then recorded;
    transient server errors are retried with backoff then raise TransientExhaustedError."""
    p = ctx.params
    temperature, seed = p.temperature, p.seed
    transient = 0
    model_attempts = 0
    t_start = time.monotonic()
    while True:
        try:
            resp = await call_once(ctx, unit, temperature, seed)
        except (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError) as e:
            # APITimeoutError is a subclass of APIConnectionError
            transient += 1
            if transient > p.transient_retries:
                raise TransientExhaustedError(f"{type(e).__name__}: {e}") from e
            delay = min(60.0, 2.0**transient) * (0.5 + random.random())
            log.debug("unit %s: %s, retry %d in %.1fs", unit.id, type(e).__name__, transient, delay)
            await asyncio.sleep(delay)
            continue
        except (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
        ) as e:
            raise FatalClientError(f"{type(e).__name__}: {e}") from e
        except (openai.BadRequestError, openai.UnprocessableEntityError) as e:
            return _record(
                ctx,
                unit,
                "request_error",
                None,
                None,
                None,
                "",
                None,
                t_start,
                model_attempts + 1,
                f"{type(e).__name__}: {e}",
            )

        model_attempts += 1
        choice = resp.choices[0]
        content = choice.message.content
        status, parsed, error = evaluate(content, choice.finish_reason, ctx.template.model)
        if status == STATUS_OK or model_attempts > p.model_retries:
            usage = resp.usage.model_dump() if resp.usage else None
            return _record(
                ctx,
                unit,
                status,
                parsed,
                content,
                choice.finish_reason,
                read_reasoning(choice.message),
                usage,
                t_start,
                model_attempts,
                error,
            )
        log.debug("unit %s: %s, model retry %d", unit.id, status, model_attempts)
        seed = p.seed + model_attempts
        temperature = max(temperature, p.retry_temperature)


def _record(
    ctx: RunContext,
    unit: WorkUnit,
    status: str,
    parsed: Any,
    raw: str | None,
    finish_reason: str | None,
    reasoning: str,
    usage: dict | None,
    t_start: float,
    attempts: int,
    error: str | None,
) -> Record:
    return Record(
        id=unit.id,
        keys=list(unit.keys),
        members=list(unit.members),
        text=unit.text,
        status=status,
        attempts=attempts,
        parsed=parsed,
        raw=None if status == STATUS_OK else raw,  # parsed already holds the content when ok
        finish_reason=finish_reason,
        reasoning_chars=len(reasoning),
        reasoning=reasoning if ctx.keep_reasoning else None,
        usage=usage,
        latency_s=round(time.monotonic() - t_start, 3),
        error=error,
        model=ctx.model_name,
        template=ctx.template.name,
        template_hash=ctx.template.hash,
        vllm_version=ctx.vllm_version,
        job_id=ctx.job_id,
    )


@dataclass
class RunOutcome:
    written: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    transient_exhausted: int = 0
    unexpected: int = 0
    not_started: int = 0
    stopped: bool = False  # stop event set (signal, breaker, fatal)
    breaker: bool = False
    fatal: str | None = None


async def run_extraction(
    ctx: RunContext,
    units: list[WorkUnit],
    writer: ResultsWriter,
    *,
    concurrency: int,
    stop: asyncio.Event,
    on_record=None,
    drain_timeout_s: float = 120.0,
    breaker_threshold: int = 8,
) -> RunOutcome:
    """Process units with `concurrency` workers. After `stop` is set no new unit starts; in-flight
    requests get drain_timeout_s to finish, then are cancelled (nothing partial is written)."""
    queue: asyncio.Queue[WorkUnit] = asyncio.Queue()
    for u in units:
        queue.put_nowait(u)
    out = RunOutcome()
    consecutive_transient = 0

    async def worker() -> None:
        nonlocal consecutive_transient
        while not stop.is_set():
            try:
                unit = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                rec = await extract_one(ctx, unit)
            except TransientExhaustedError as e:
                out.transient_exhausted += 1
                consecutive_transient += 1
                log.warning("unit %s not done (server unreachable): %s", unit.id, e)
                if consecutive_transient >= breaker_threshold and not stop.is_set():
                    log.error(
                        "circuit breaker: %d consecutive units failed to reach the server",
                        consecutive_transient,
                    )
                    out.breaker = True
                    stop.set()
                continue
            except FatalClientError as e:
                log.error("fatal client error, stopping: %s", e)
                out.fatal = str(e)
                stop.set()
                return
            except Exception:  # bug or unexpected response: keep the run going, unit redone later
                out.unexpected += 1
                log.exception("unit %s: unexpected error (not recorded)", unit.id)
                continue
            consecutive_transient = 0
            writer.write(rec)
            out.written += 1
            out.by_status[rec.status] = out.by_status.get(rec.status, 0) + 1
            if on_record:
                on_record(rec)

    n_workers = max(1, min(concurrency, len(units)))
    tasks = [asyncio.create_task(worker()) for _ in range(n_workers)]
    all_done = asyncio.gather(*tasks)
    stop_wait = asyncio.create_task(stop.wait())
    await asyncio.wait({all_done, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
    if not all_done.done():
        log.warning("stop requested: draining in-flight requests (up to %.0fs)", drain_timeout_s)
        try:
            await asyncio.wait_for(asyncio.shield(all_done), drain_timeout_s)
        except TimeoutError:
            log.warning("drain timeout: cancelling remaining requests")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    stop_wait.cancel()
    await asyncio.gather(stop_wait, return_exceptions=True)
    # Workers catch their own errors; retrieve the gather result so nothing is left unobserved.
    await asyncio.gather(all_done, return_exceptions=True)
    out.stopped = stop.is_set()
    out.not_started = queue.qsize()
    return out
