"""Paths shared by the bash scripts and the Python code (configs/paths.env)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / "configs" / "paths.env"

_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _expand(value: str, env: dict[str, str]) -> str:
    """Expand $VAR, ${VAR} and ${VAR:-default} like bash; defaults may nest (${A:-${B}/x})."""
    out: list[str] = []
    i = 0
    while i < len(value):
        c = value[i]
        if c != "$":
            out.append(c)
            i += 1
            continue
        if value.startswith("${", i):
            depth, j = 1, i + 2
            while j < len(value) and depth:
                if value.startswith("${", j):
                    depth += 1
                    j += 2
                    continue
                if value[j] == "}":
                    depth -= 1
                j += 1
            if depth:
                raise ValueError(f"unbalanced braces in {value!r}")
            inner = value[i + 2 : j - 1]
            name, sep, default = inner.partition(":-")
            if not _NAME.fullmatch(name):
                raise ValueError(f"unsupported expansion ${{{inner}}}")
            current = env.get(name, "")
            out.append(_expand(default, env) if sep and current == "" else current)
            i = j
            continue
        m = _NAME.match(value, i + 1)
        if m:
            out.append(env.get(m.group(0), ""))
            i = m.end()
        else:
            out.append(c)
            i += 1
    return "".join(out)


def parse_env_file(path: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Parse a KEY=VALUE file the way `set -a; source file` would (subset of bash syntax)."""
    env = dict(os.environ if environ is None else environ)
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"{path}: cannot parse line: {raw!r}")
        key, value = m.group(1), m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        value = _expand(value, env)
        env[key] = value
        values[key] = value
    return values


@dataclass(frozen=True)
class Paths:
    root: Path
    python: Path
    data: Path
    models_dir: Path
    containers_dir: Path
    tiktoken_dir: Path
    vllm_image: str
    sif: Path
    runs_dir: Path
    scratch: Path
    proxy_script: Path
    vllm_cache: Path | None
    hnvme_max_files: int

    @property
    def templates_dir(self) -> Path:
        return self.root / "templates"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"


def load_paths(env_file: str | Path | None = None) -> Paths:
    """Load configs/paths.env (or $RSX_PATHS); variables already in the environment take precedence."""
    path = Path(env_file or os.environ.get("RSX_PATHS") or DEFAULT_ENV_FILE)
    v = parse_env_file(path)
    cache = v.get("RSX_VLLM_CACHE", "")
    return Paths(
        root=Path(v.get("RSX_ROOT") or REPO_ROOT),
        python=Path(v["PYTHON"]),
        data=Path(v["RSX_DATA"]),
        models_dir=Path(v["RSX_MODELS_DIR"]),
        containers_dir=Path(v["RSX_CONTAINERS_DIR"]),
        tiktoken_dir=Path(v["RSX_TIKTOKEN_DIR"]),
        vllm_image=v["RSX_VLLM_IMAGE"],
        sif=Path(v["RSX_SIF"]),
        runs_dir=Path(v["RSX_RUNS_DIR"]),
        scratch=Path(v["RSX_SCRATCH"]),
        proxy_script=Path(v["RSX_PROXY_SCRIPT"]),
        vllm_cache=Path(cache) if cache else None,
        hnvme_max_files=int(v.get("RSX_HNVME_MAX_FILES") or 55000),
    )
