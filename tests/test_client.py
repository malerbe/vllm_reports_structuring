"""Client tests against a fake OpenAI-compatible server (httpx.MockTransport, no network)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import openai
import pytest
from conftest import valid_report

from rsx.client import RunContext, evaluate, run_extraction
from rsx.data import WorkUnit
from rsx.registry import RequestDefaults
from rsx.results import ResultsWriter, latest_records
from rsx.template import load_template


def completion(
    content: str | None, finish_reason: str = "stop", reasoning: str = "thinking"
) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, "reasoning": reasoning},
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


class FakeServer:
    """Scripted responses per unit text; records the max number of concurrent requests."""

    def __init__(self, script):
        self.script = script  # callable(text, attempt) -> (status_code, body) or Exception
        self.calls: dict[str, int] = {}
        self.inflight = 0
        self.max_inflight = 0
        self.bodies: list[dict] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        text = body["messages"][1]["content"]
        self.calls[text] = self.calls.get(text, 0) + 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(0.01)
            result = self.script(text, self.calls[text])
            if isinstance(result, Exception):
                raise result
            code, payload = result
            return httpx.Response(code, json=payload)
        finally:
            self.inflight -= 1


def make_ctx(template, server: FakeServer, **params) -> RunContext:
    http = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    client = openai.AsyncOpenAI(
        base_url="http://test/v1", api_key="k", max_retries=0, http_client=http
    )
    defaults = {"transient_retries": 2, "model_retries": 1, **params}
    return RunContext(
        client=client,
        template=template,
        served_name="srv",
        model_name="m",
        params=RequestDefaults(**defaults),
        vllm_version="0.31.0",
        job_id="1",
    )


def run(ctx, units, path: Path, concurrency=4, stop=None, **kw):
    async def main():
        with ResultsWriter(path / "results.jsonl") as w:
            return await run_extraction(
                ctx, units, w, concurrency=concurrency, stop=stop or asyncio.Event(), **kw
            )

    return asyncio.run(main())


def units(n: int) -> list[WorkUnit]:
    return [
        WorkUnit(id=f"{i:016x}", text=f"report {i}", members=(f"m{i}",), keys=(f"k{i}",))
        for i in range(n)
    ]


@pytest.fixture
def template(typed_template_dir):
    return load_template(typed_template_dir)


def test_ok_and_request_shape(template, tmp_path):
    server = FakeServer(lambda text, n: (200, completion(json.dumps(valid_report()))))
    out = run(
        make_ctx(template, server, reasoning_effort="low", temperature=0.0), units(3), tmp_path
    )
    assert out.by_status == {"ok": 3} and out.written == 3
    body = server.bodies[0]
    assert body["model"] == "srv"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == template.json_schema
    assert body["reasoning_effort"] == "low" and body["temperature"] == 0.0
    assert body["max_completion_tokens"] == 8192
    recs = latest_records(tmp_path)
    r = next(iter(recs.values()))
    assert r.status == "ok" and r.parsed == valid_report() and r.raw is None
    assert r.reasoning_chars == len("thinking") and r.reasoning is None
    assert r.template_hash == template.hash and r.vllm_version == "0.31.0"
    assert r.usage["completion_tokens"] == 50


def test_model_failure_retried_then_recorded(template, tmp_path):
    server = FakeServer(lambda text, n: (200, completion("{not json")))
    out = run(make_ctx(template, server, model_retries=1), units(1), tmp_path)
    assert out.by_status == {"invalid_json": 1}
    r = next(iter(latest_records(tmp_path).values()))
    assert r.attempts == 2 and r.raw == "{not json" and "JSONDecodeError" in r.error
    temps = [b["temperature"] for b in server.bodies]
    seeds = [b["seed"] for b in server.bodies]
    assert temps == [0.0, 0.7] and seeds == [0, 1]


def test_retry_then_success(template, tmp_path):
    def script(text, n):
        return (
            (200, completion("{bad")) if n == 1 else (200, completion(json.dumps(valid_report())))
        )

    out = run(make_ctx(template, FakeServer(script)), units(1), tmp_path)
    assert out.by_status == {"ok": 1}


def test_truncated_and_validation_error(template, tmp_path):
    bad = valid_report()
    bad["findings"][0]["kind"] = "tumour"

    def script(text, n):
        if text.endswith("0\n"):
            return 200, completion('{"exam": {', finish_reason="length")
        return 200, completion(json.dumps(bad))

    out = run(make_ctx(template, FakeServer(script), model_retries=0), units(2), tmp_path)
    assert out.by_status == {"truncated": 1, "validation_error": 1}


def test_transient_errors_retried(template, tmp_path, monkeypatch):
    monkeypatch.setattr("rsx.client.random.random", lambda: 0.0)
    monkeypatch.setattr("rsx.client.asyncio.sleep", _fast_sleep)

    def script(text, n):
        return (500, {"error": "boom"}) if n == 1 else (200, completion(json.dumps(valid_report())))

    out = run(make_ctx(template, FakeServer(script)), units(1), tmp_path)
    assert out.by_status == {"ok": 1}


def test_unreachable_server_writes_nothing_and_trips_breaker(template, tmp_path, monkeypatch):
    monkeypatch.setattr("rsx.client.asyncio.sleep", _fast_sleep)
    server = FakeServer(lambda text, n: httpx.ConnectError("refused"))
    out = run(
        make_ctx(template, server, transient_retries=1),
        units(20),
        tmp_path,
        concurrency=2,
        breaker_threshold=3,
    )
    assert out.written == 0 and out.breaker and out.stopped
    assert out.transient_exhausted >= 3 and out.not_started > 0
    assert latest_records(tmp_path) == {}


def test_bad_request_recorded(template, tmp_path):
    server = FakeServer(lambda text, n: (400, {"error": {"message": "prompt too long"}}))
    out = run(make_ctx(template, server), units(1), tmp_path)
    assert out.by_status == {"request_error": 1}
    assert len(server.bodies) == 1


def test_fatal_error_stops(template, tmp_path):
    server = FakeServer(lambda text, n: (404, {"error": {"message": "model not found"}}))
    out = run(make_ctx(template, server), units(10), tmp_path, concurrency=1)
    assert out.fatal and out.written == 0 and out.not_started == 9


def test_concurrency_bounded(template, tmp_path):
    server = FakeServer(lambda text, n: (200, completion(json.dumps(valid_report()))))
    out = run(make_ctx(template, server), units(30), tmp_path, concurrency=5)
    assert out.written == 30 and 1 < server.max_inflight <= 5


def test_stop_drains_inflight(template, tmp_path):
    stop = asyncio.Event()
    server = FakeServer(lambda text, n: (200, completion(json.dumps(valid_report()))))
    seen = []

    def on_record(rec):
        seen.append(rec.id)
        if len(seen) == 3:
            stop.set()

    out = run(
        make_ctx(template, server),
        units(50),
        tmp_path,
        concurrency=2,
        stop=stop,
        on_record=on_record,
    )
    assert out.stopped and 3 <= out.written < 50 and out.not_started > 0
    assert (
        len(latest_records(tmp_path)) == out.written
    )  # every started request was completed and written


def test_evaluate():
    from rsx.template import build_model, parse_instructions

    model = build_model(parse_instructions({"a": {"_type": "integer"}}))
    assert evaluate('{"a": 1}', "stop", model)[0] == "ok"
    assert evaluate('{"a": "x"}', "stop", model)[0] == "validation_error"
    assert evaluate("", "stop", model)[0] == "empty"
    assert evaluate('{"a": 1}', "length", model)[0] == "truncated"


_real_sleep = asyncio.sleep


async def _fast_sleep(delay, *args, **kwargs):
    await _real_sleep(0)
