"""Download model weights and tiktoken files (login node, with internet).

Run through scripts/download_models.sh, which sets the proxy and the HF cache location.
Only needs pyyaml, pydantic and huggingface_hub, so it works before `pip install -e .`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.request
from pathlib import Path

from .paths import load_paths
from .registry import ModelSpec, load_registry

log = logging.getLogger("rsx.download")

# oss-harmony (vLLM >= 0.30) embeds these vocabularies; they are kept as an offline fallback
# (TIKTOKEN_ENCODINGS_BASE). SHA-256 values as checked by harmony.
TIKTOKEN_FILES = {
    "o200k_base.tiktoken": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    "cl100k_base.tiktoken": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
}
TIKTOKEN_URL = "https://openaipublic.blob.core.windows.net/encodings/{name}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_model_dir(path: Path) -> list[str]:
    """Return a list of problems (empty = complete): config, tokenizer, every indexed shard."""
    problems = []
    for name in ("config.json", "tokenizer.json"):
        if not (path / name).is_file():
            problems.append(f"missing {name}")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        problems += [f"missing shard {s}" for s in shards if not (path / s).is_file()]
    elif not list(path.glob("*.safetensors")):
        problems.append("no *.safetensors files")
    return problems


def download_model(spec: ModelSpec, models_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    target = spec.model_path(models_dir)
    if target.is_dir() and not check_model_dir(target):
        log.info("%s: already complete in %s", spec.name, target)
        return target
    log.info("%s: downloading %s -> %s (exclude %s)", spec.name, spec.hf_repo, target, spec.hf_exclude)
    snapshot_download(
        repo_id=spec.hf_repo,
        local_dir=str(target),
        ignore_patterns=spec.hf_exclude or None,
    )
    problems = check_model_dir(target)
    if problems:
        raise RuntimeError(f"{spec.name}: incomplete download in {target}: {problems}")
    log.info("%s: complete", spec.name)
    return target


def download_tiktoken(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, digest in TIKTOKEN_FILES.items():
        path = dest / name
        if path.is_file() and sha256(path) == digest:
            continue
        tmp = path.with_suffix(".partial")
        log.info("tiktoken: fetching %s", name)
        with urllib.request.urlopen(TIKTOKEN_URL.format(name=name), timeout=120) as r:
            tmp.write_bytes(r.read())
        got = sha256(tmp)
        if got != digest:
            tmp.unlink()
            raise RuntimeError(f"tiktoken {name}: sha256 mismatch ({got})")
        tmp.replace(path)
    log.info("tiktoken: files OK in %s", dest)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", nargs="*", help="model names from configs/models.yaml (default: all)")
    p.add_argument("--check", action="store_true", help="only check completeness, do not download")
    p.add_argument("--no-tiktoken", action="store_true")
    args = p.parse_args(argv)

    paths = load_paths()
    registry = load_registry()
    names = args.models or list(registry)
    unknown = [n for n in names if n not in registry]
    if unknown:
        p.error(f"unknown model(s) {unknown}; known: {sorted(registry)}")

    status = 0
    for name in names:
        spec = registry[name]
        if args.check:
            problems = check_model_dir(spec.model_path(paths.models_dir))
            print(f"{name}: {'OK' if not problems else problems}")
            status |= bool(problems)
        else:
            download_model(spec, paths.models_dir)
    if not args.check and not args.no_tiktoken:
        download_tiktoken(paths.tiktoken_dir)
    return status


if __name__ == "__main__":
    sys.exit(main())
