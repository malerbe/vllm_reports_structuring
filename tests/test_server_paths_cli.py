from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

import pytest
from conftest import REPO

from rsx.cli import main
from rsx.paths import _expand, load_paths, parse_env_file
from rsx.registry import ModelSpec, RequestDefaults, load_registry
from rsx.server import (
    ServerDiedError,
    ServerTimeoutError,
    VllmServer,
    build_command,
    pick_port,
    vllm_flags,
)


def test_env_expansion():
    env = {"A": "/a", "EMPTY": ""}
    assert _expand("${A}/x", env) == "/a/x"
    assert _expand("${B:-${A}/d}", env) == "/a/d"
    assert _expand("${A:-other}", env) == "/a"
    assert _expand("${EMPTY:-${A:-z}/m}", env) == "/a/m"
    assert _expand("$A-$B", env) == "/a-"


def test_paths_env_file(tmp_path: Path):
    f = tmp_path / "p.env"
    f.write_text("# c\nX=${X:-/base}\nY=${Y:-${X}/sub}\nQ='quoted'\n")
    assert parse_env_file(f, {}) == {"X": "/base", "Y": "/base/sub", "Q": "quoted"}
    assert parse_env_file(f, {"X": "/over"})["Y"] == "/over/sub"


def test_repo_paths_and_registry():
    p = load_paths()
    assert p.sif.name.endswith(".sif") and "}" not in str(p.tiktoken_dir)
    reg = load_registry()
    assert {"gpt-oss-120b", "gpt-oss-20b"} <= set(reg)
    assert reg["gpt-oss-20b"].request.reasoning_effort == "low"


def test_request_merge():
    r = RequestDefaults().merged({"temperature": 0.5, "max_tokens": None})
    assert r.temperature == 0.5 and r.max_tokens == 8192


def spec(**kw) -> ModelSpec:
    return ModelSpec(
        name="m",
        hf_repo="o/m",
        local_dir="m",
        served_name="srv",
        vllm_args={
            "max-model-len": 1024,
            "async-scheduling": True,
            "x": False,
            "structured-outputs-config": {"backend": "xgrammar"},
        },
        **kw,
    )


def test_vllm_flags():
    assert vllm_flags({"a": 1, "b": True, "c": False, "d": None, "e": {"k": 1}}) == [
        "--a",
        "1",
        "--b",
        "--e",
        '{"k": 1}',
    ]


def test_build_command(tmp_path: Path):
    paths = load_paths()
    cmd = build_command(paths, spec(), 23456, tmp_path)
    assert cmd[:4] == ["apptainer", "exec", "--nv", "--cleanenv"]
    assert f"{paths.models_dir / 'm'}:/rsx_model:ro" in cmd
    i = cmd.index("serve")
    assert cmd[i + 1] == "/rsx_model" and "--async-scheduling" in cmd and "--x" not in cmd
    assert cmd[cmd.index("--port") + 1] == "23456" and cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert not any("api-key" in c for c in cmd)  # passed through the environment, not argv


def test_pick_port_skips_busy():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        busy = s.getsockname()[1]
        job_id = str(busy - 20000) if busy >= 20000 else None
        if job_id is None:
            pytest.skip("ephemeral port below 20000")
        assert pick_port(job_id) != busy


def test_wait_ready_detects_dead_server(tmp_path: Path):
    srv = VllmServer(load_paths(), spec(), port=1, cache_dir=tmp_path)
    srv.proc = subprocess.Popen(["false"])
    srv._t0 = 0.0
    srv.proc.wait()
    with pytest.raises(ServerDiedError):
        srv.wait_ready(timeout_s=5, poll_s=0.1)


def test_wait_ready_timeout(tmp_path: Path):
    srv = VllmServer(load_paths(), spec(), port=pick_port(None), cache_dir=tmp_path)
    srv._t0 = 0.0
    with pytest.raises(ServerTimeoutError):
        srv.wait_ready(timeout_s=0.3, poll_s=0.1)


def test_cli_template_check(capsys):
    assert main(["template", "check", "chest_ct_findings"]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_template_check_radfields_file(capsys):
    path = REPO.parent / "labels_extraction/RadFields/templates/de/ct/ct_thorax_standard.yaml"
    if not path.exists():
        pytest.skip("RadFields checkout not found")
    assert main(["template", "check", str(path)]) == 0


def test_cli_dry_run(ct_rate_csv: Path, typed_template_dir: Path, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("RSX_RUNS_DIR", str(tmp_path / "runs"))
    assert (
        main(
            [
                "run",
                "--dry-run",
                "-t",
                str(typed_template_dir),
                "-m",
                "gpt-oss-20b",
                "-i",
                str(ct_rate_csv),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "4 units" in out and "----- system -----" in out


def test_cli_submit_print_only(ct_rate_csv: Path, capsys):
    assert (
        main(
            [
                "submit",
                "-t",
                "chest_ct_findings",
                "-m",
                "gpt-oss-20b",
                "-i",
                str(ct_rate_csv),
                "-p",
                "preempt",
                "--print-only",
                "--",
                "--limit",
                "3",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "sbatch --partition preempt" in out and "--limit 3" in out and "run.sbatch job" in out


def test_cli_feeders_and_models(capsys):
    assert main(["feeders"]) == 0
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "ct_rate" in out and "gpt-oss-120b" in out


def test_stats_json(tmp_path: Path, capsys):
    from rsx.results import Record, ResultsWriter

    with ResultsWriter(tmp_path / "results.jsonl") as w:
        w.write(
            Record(
                id="a",
                keys=["a"],
                members=["a"],
                text="t",
                status="ok",
                latency_s=2.0,
                usage={"prompt_tokens": 3, "completion_tokens": 4},
            )
        )
    assert main(["stats", str(tmp_path)]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["status"] == {"ok": 1} and s["completion_tokens"]["total"] == 4
