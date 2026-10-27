"""Regression tests for defects found in the code review."""

from __future__ import annotations

import asyncio
import json
import signal
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import yaml
from conftest import valid_report

from rsx.cli import main
from rsx.job import _shards, _StopFlag
from rsx.results import Record, ResultsWriter
from rsx.template import build_model, load_template, parse_instructions, vllm_json_schema


def test_fields_named_like_schema_annotations_are_kept(tmp_path: Path):
    raw = {
        "title": "report title",
        "default": {"_type": "boolean"},
        "items": [{"description": "lesion description", "title": {"_type": "string"}}],
    }
    model = build_model(parse_instructions(raw))
    schema = vllm_json_schema(model)
    assert list(schema["properties"]) == ["title", "default", "items"]
    assert schema["required"] == ["title", "default", "items"]
    assert list(schema["properties"]["items"]["items"]["properties"]) == ["description", "title"]
    obj = {"title": "x", "default": None, "items": [{"description": "y", "title": None}]}
    model.model_validate(obj)


def test_single_value_enum_is_portable(tmp_path: Path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "template.yaml").write_text(
        "kind: {_type: enum, values: [present]}\ntags:\n  - {_type: enum, values: [only]}\n"
    )
    t = load_template(d)
    kind = t.json_schema["properties"]["kind"]
    assert {"enum": ["present"], "type": "string"} in kind["anyOf"]
    assert t.json_schema["properties"]["tags"]["items"] == {"enum": ["only"], "type": "string"}


def test_snapshot_with_custom_postprocess_name(tmp_path: Path, typed_template_dir: Path):
    (typed_template_dir / "size_hook.py").write_text("def postprocess(table, df):\n    return df\n")
    cfg = yaml.safe_load((typed_template_dir / "config.yaml").read_text())
    cfg["postprocess"] = "size_hook.py"
    (typed_template_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    t = load_template(typed_template_dir)
    t.snapshot(tmp_path / "snap")
    again = load_template(tmp_path / "snap")
    assert again.postprocess_path().name == "postprocess.py"


def _args(**kw):
    return Namespace(**{"num_shards": None, "shard_id": None, **kw})


@pytest.mark.parametrize(
    "env, kw, expected",
    [
        ({}, {}, (1, 0)),
        (
            {
                "SLURM_ARRAY_TASK_ID": "2",
                "SLURM_ARRAY_TASK_COUNT": "4",
                "SLURM_ARRAY_TASK_MIN": "0",
                "SLURM_ARRAY_TASK_MAX": "3",
            },
            {},
            (4, 2),
        ),
        # explicit --num-shards inside an array: the array index is the shard id
        (
            {
                "SLURM_ARRAY_TASK_ID": "3",
                "SLURM_ARRAY_TASK_COUNT": "4",
                "SLURM_ARRAY_TASK_MIN": "0",
                "SLURM_ARRAY_TASK_MAX": "3",
            },
            {"num_shards": 4},
            (4, 3),
        ),
        # rerun of one shard: sbatch --array=2 ... --num-shards 4
        (
            {
                "SLURM_ARRAY_TASK_ID": "2",
                "SLURM_ARRAY_TASK_COUNT": "1",
                "SLURM_ARRAY_TASK_MIN": "2",
                "SLURM_ARRAY_TASK_MAX": "2",
            },
            {"num_shards": 4},
            (4, 2),
        ),
        ({}, {"num_shards": 3, "shard_id": 1}, (3, 1)),
    ],
)
def test_shards(monkeypatch, env, kw, expected):
    for k in (
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_MIN",
        "SLURM_ARRAY_TASK_MAX",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert _shards(_args(**kw)) == expected


def test_shards_rejects_ambiguous(monkeypatch):
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "2")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MAX", "2")
    with pytest.raises(ValueError, match="num-shards"):
        _shards(_args())
    with pytest.raises(ValueError, match="needs --num-shards"):
        _shards(_args(shard_id=1))


def test_signal_after_loop_closed_is_harmless():
    stop = _StopFlag()

    async def main_():
        stop.event, stop.loop = asyncio.Event(), asyncio.get_running_loop()

    asyncio.run(main_())  # loop now closed, references still set
    stop.trigger(signal.SIGTERM)
    assert stop()


def test_export_column_clash_renamed(tmp_path: Path):
    d = tmp_path / "tpl"
    d.mkdir()
    (d / "template.yaml").write_text("status: {_type: string}\nitems:\n  - unit_id: x\n")
    (d / "config.yaml").write_text("feeder: ct_rate\nexport_tables: {items: items}\n")
    t = load_template(d)
    run = tmp_path / "run"
    t.snapshot(run / "template")
    (run / "run.yaml").write_text(yaml.safe_dump({"member_column": "id", "template": "tpl"}))
    with ResultsWriter(run / "results.jsonl") as w:
        w.write(
            Record(
                id="u",
                keys=["k"],
                members=["m"],
                text="t",
                status="ok",
                parsed={"status": "post-op", "items": [{"unit_id": "z"}]},
                template_hash=t.hash,
            )
        )
    from rsx.export import export_run

    written = export_run(run)
    reports = pd.read_parquet(written["reports.parquet"])
    assert reports.loc[0, "status"] == "ok" and reports.loc[0, "status__field"] == "post-op"
    items = pd.read_parquet(written["items.parquet"])
    assert items.loc[0, "unit_id"] == "u" and items.loc[0, "unit_id__field"] == "z"
    summary = json.loads(written["summary.json"].read_text())
    assert summary["template"] == "tpl"


def test_write_failure_stops_run(typed_template_dir: Path, tmp_path: Path):
    import httpx
    import openai

    from rsx.client import RunContext, run_extraction
    from rsx.data import WorkUnit
    from rsx.registry import RequestDefaults

    body = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(valid_report())},
            }
        ],
    }

    async def handler(request):
        return httpx.Response(200, json=body)

    class FailingWriter:
        def write(self, rec):
            raise OSError(122, "Disk quota exceeded")

    async def go():
        client = openai.AsyncOpenAI(
            base_url="http://t/v1",
            api_key="k",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        ctx = RunContext(
            client=client,
            template=load_template(typed_template_dir),
            served_name="s",
            model_name="m",
            params=RequestDefaults(),
        )
        units = [
            WorkUnit(id=f"{i:016x}", text=f"r{i}", members=(str(i),), keys=(str(i),))
            for i in range(10)
        ]
        return await run_extraction(
            ctx, units, FailingWriter(), concurrency=2, stop=asyncio.Event()
        )

    out = asyncio.run(go())
    assert out.fatal and "quota" in out.fatal and out.stopped and out.written == 0


def test_submit_resolves_template_to_absolute(
    ct_rate_csv: Path, typed_template_dir: Path, capsys, monkeypatch
):
    monkeypatch.chdir(typed_template_dir.parent)
    assert (
        main(
            [
                "submit",
                "-t",
                f"./{typed_template_dir.name}",
                "-m",
                "gpt-oss-20b",
                "-i",
                str(ct_rate_csv),
                "--array",
                "2",
                "--print-only",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert f"--template {typed_template_dir}" in out
    assert "%x_%A_%a.log" in out and "--array 0-1" in out


def test_submit_rejects_missing_template(ct_rate_csv: Path):
    with pytest.raises(FileNotFoundError):
        main(["submit", "-t", "no_such_template", "-i", str(ct_rate_csv), "--print-only"])
