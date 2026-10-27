"""vLLM server lifecycle: `vllm serve` inside the Apptainer image, on 127.0.0.1, one per job."""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from .paths import Paths
from .registry import ModelSpec

log = logging.getLogger(__name__)

MODEL_MOUNT = "/rsx_model"
TIKTOKEN_MOUNT = "/rsx_tiktoken"
CACHE_MOUNT = "/rsx_cache"


class ServerDiedError(RuntimeError):
    pass


class ServerTimeoutError(RuntimeError):
    pass


def pick_port(job_id: str | None = None, host: str = "127.0.0.1") -> int:
    """Port derived from the job id (h200 nodes are shared between jobs), then probed until free."""
    base = 20000 + (
        int(job_id) % 20000 if job_id and job_id.isdigit() else secrets.randbelow(20000)
    )
    for offset in range(200):
        port = 20000 + (base - 20000 + offset) % 20000
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port found")


def vllm_flags(args: dict[str, Any]) -> list[str]:
    """{flag: value} -> CLI flags. true -> bare flag; false/None -> omitted; dict/list -> JSON."""
    out: list[str] = []
    for key, value in args.items():
        flag = key if key.startswith("--") else f"--{key}"
        if value is True:
            out.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, (dict, list)):
            out += [flag, json.dumps(value)]
        else:
            out += [flag, str(value)]
    return out


def container_env(paths: Paths, spec: ModelSpec, api_key: str) -> dict[str, str]:
    """Environment inside the container (passed as APPTAINERENV_*: works with --cleanenv and
    keeps the API key out of the command line visible in `ps`)."""
    env = {
        "VLLM_API_KEY": api_key,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_DO_NOT_TRACK": "1",
        "DO_NOT_TRACK": "1",
        "PYTHONNOUSERSITE": "1",
        # gpt-oss: harmony puts the current date in the system message; a fixed date keeps
        # requests deterministic and the prompt prefix cacheable.
        "VLLM_SYSTEM_START_DATE": os.environ.get("RSX_SYSTEM_DATE", "2026-01-01"),
        "XDG_CACHE_HOME": CACHE_MOUNT,
        "VLLM_CACHE_ROOT": f"{CACHE_MOUNT}/vllm",
        "TRITON_CACHE_DIR": f"{CACHE_MOUNT}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{CACHE_MOUNT}/inductor",
        "CUDA_CACHE_PATH": f"{CACHE_MOUNT}/nv",
        "TMPDIR": f"{CACHE_MOUNT}/tmp",
    }
    if paths.tiktoken_dir.is_dir():
        env["TIKTOKEN_ENCODINGS_BASE"] = TIKTOKEN_MOUNT
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    env.update({k: str(v) for k, v in spec.vllm_env.items()})
    return env


def build_command(
    paths: Paths, spec: ModelSpec, port: int, cache_dir: Path, host: str = "127.0.0.1"
) -> list[str]:
    model_dir = spec.model_path(paths.models_dir)
    cmd = [
        "apptainer",
        "exec",
        "--nv",
        "--cleanenv",
        "--bind",
        f"{model_dir}:{MODEL_MOUNT}:ro",
        "--bind",
        f"{cache_dir}:{CACHE_MOUNT}",
    ]
    if paths.tiktoken_dir.is_dir():
        cmd += ["--bind", f"{paths.tiktoken_dir}:{TIKTOKEN_MOUNT}:ro"]
    cmd += [
        str(paths.sif),
        "vllm",
        "serve",
        MODEL_MOUNT,
        "--served-model-name",
        spec.served_name,
        "--host",
        host,
        "--port",
        str(port),
    ]
    return cmd + vllm_flags(spec.vllm_args)


def default_cache_dir(paths: Paths, job_id: str | None) -> Path:
    if paths.vllm_cache:
        return paths.vllm_cache
    base = Path(os.environ.get("TMPDIR") or "/tmp")
    return base / f"rsx_{job_id or os.getpid()}"


class VllmServer:
    """Start `vllm serve` in the background, wait for /health, stop the whole process group."""

    def __init__(
        self,
        paths: Paths,
        spec: ModelSpec,
        *,
        job_id: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        host: str = "127.0.0.1",
    ):
        self.paths, self.spec, self.host = paths, spec, host
        self.job_id = job_id
        self.port = port or pick_port(job_id, host)
        self.api_key = api_key or secrets.token_hex(16)
        self.cache_dir = cache_dir or default_cache_dir(paths, job_id)
        self._owns_cache = not paths.vllm_cache and cache_dir is None
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.startup_s: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def root_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def check_files(self) -> None:
        if not self.paths.sif.is_file():
            raise FileNotFoundError(
                f"vLLM image {self.paths.sif} missing: run scripts/pull_vllm_image.sh"
            )
        model_dir = self.spec.model_path(self.paths.models_dir)
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"model {model_dir} missing: run scripts/download_models.sh {self.spec.name}"
            )

    def start(self, log_path: Path) -> None:
        self.check_files()
        for sub in ("vllm", "triton", "inductor", "nv", "tmp"):
            (self.cache_dir / sub).mkdir(parents=True, exist_ok=True)
        cmd = build_command(self.paths, self.spec, self.port, self.cache_dir, self.host)
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("APPTAINERENV_", "SINGULARITYENV_"))
        }
        env.update(
            {
                f"APPTAINERENV_{k}": v
                for k, v in container_env(self.paths, self.spec, self.api_key).items()
            }
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        log.info("starting vLLM (%s) on %s, log %s", self.spec.name, self.root_url, log_path)
        log.info("command: %s", " ".join(cmd))
        self._t0 = time.monotonic()
        with log_path.open("ab") as f:
            self.proc = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )

    def log_tail(self, n: int = 40) -> str:
        if not self.log_path or not self.log_path.exists():
            return ""
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])

    def wait_ready(self, timeout_s: float | None = None, poll_s: float = 5.0, stop=None) -> float:
        """Block until GET /health returns 200. Raises ServerDiedError / ServerTimeoutError."""
        timeout_s = timeout_s or self.spec.startup_timeout_s
        deadline = time.monotonic() + timeout_s
        last_log = 0.0
        with httpx.Client(trust_env=False, timeout=5.0) as http:
            while True:
                if self.proc is not None and self.proc.poll() is not None:
                    raise ServerDiedError(
                        f"vLLM exited with code {self.proc.returncode}. "
                        f"Log tail:\n{self.log_tail()}"
                    )
                if stop is not None and stop():
                    raise ServerTimeoutError("interrupted while waiting for the server")
                try:
                    if http.get(f"{self.root_url}/health").status_code == 200:
                        self.startup_s = time.monotonic() - self._t0
                        log.info("server ready after %.0fs", self.startup_s)
                        return self.startup_s
                except httpx.HTTPError:
                    pass
                now = time.monotonic()
                if now > deadline:
                    raise ServerTimeoutError(
                        f"server not ready after {timeout_s:.0f}s. Log tail:\n{self.log_tail()}"
                    )
                if now - last_log > 60:
                    log.info("waiting for vLLM (%.0fs elapsed)", now - self._t0)
                    last_log = now
                time.sleep(poll_s)

    def info(self) -> dict[str, Any]:
        """vLLM version and served models."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        out: dict[str, Any] = {"base_url": self.base_url, "startup_s": self.startup_s}
        with httpx.Client(trust_env=False, timeout=10.0) as http:
            try:
                out["vllm_version"] = (
                    http.get(f"{self.root_url}/version", headers=headers).json().get("version")
                )
            except (httpx.HTTPError, ValueError):
                out["vllm_version"] = None
            try:
                models = http.get(f"{self.root_url}/v1/models", headers=headers).json()
                out["served_models"] = [m["id"] for m in models.get("data", [])]
            except (httpx.HTTPError, ValueError, KeyError):
                out["served_models"] = []
        if out["served_models"] and self.spec.served_name not in out["served_models"]:
            raise RuntimeError(
                f"server serves {out['served_models']}, expected {self.spec.served_name}"
            )
        return out

    def stop(self, grace_s: float = 60.0) -> None:
        """SIGTERM the process group, SIGKILL after grace_s. Idempotent."""
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            log.info("stopping vLLM")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                log.warning("vLLM did not stop in %.0fs, killing", grace_s)
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
            except ProcessLookupError:
                pass
        if self._owns_cache and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def connection_info(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.spec.name,
            "served_name": self.spec.served_name,
            "job_id": self.job_id,
            "host": socket.gethostname(),
        }
