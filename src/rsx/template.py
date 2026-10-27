"""Templates: RadFields-compatible YAML (+ typed leaves) -> Pydantic model -> vLLM JSON schema.

Template format (a superset of RadFields templates, which load unchanged):

* mapping              -> section (nested object)
* list                 -> repeated entries; the element type comes from the first item
* string               -> text field (str); the string is the hint shown to the LLM
* other scalar / null  -> text field, value converted to str / "" (as RadFields does)

Extensions (keys starting with "_" are reserved):

* a mapping containing ``_type`` is a *typed leaf*::

      size_mm: {_type: number, _hint: "largest diameter in mm"}
      status:  {_type: enum, values: [present, uncertain], _nullable: false}

  ``_type`` is one of enum, number, integer, boolean, string; ``values`` (enum only) lists the
  allowed strings; ``_hint`` is shown to the LLM; ``_nullable`` defaults to true, and null means
  "not mentioned in the report". Inside a list, a typed leaf is not nullable by default.
* a section may carry ``_hint`` and ``_nullable`` (``_nullable: true`` -> the whole object may be null)
* ``_vocab`` at the root is ignored: a place to define YAML anchors reused below.

Every property is required in the JSON schema (constrained decoding always emits every key), so
"not mentioned" is an explicit null (typed leaves) or the template's missing_value (text leaves).
"""

from __future__ import annotations

import hashlib
import json
import keyword
import re
import shutil
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, create_model

from .paths import REPO_ROOT

TYPED_KINDS = ("enum", "number", "integer", "boolean", "string")
TYPED_LEAF_KEYS = {"_type", "values", "_hint", "_nullable"}
SECTION_META_KEYS = {"_hint", "_nullable"}
ROOT_ONLY_KEYS = {"_vocab"}
# JSON-schema keywords the generated schema may contain (all supported by xgrammar / guidance).
PORTABLE_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum", "anyOf"}
# German umlauts as transliterated by RadFields (rad_fields.utils.NON_ASCII_CHARS_TRANSCRIPTION).
_TRANSLIT = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}
_BASEMODEL_ATTRS = set(dir(BaseModel))

DEFAULT_PROMPTS_DIR = REPO_ROOT / "templates" / "_default"


class TemplateError(ValueError):
    """Malformed template. The message carries the path of the offending node (e.g. $.findings[].size)."""

    def __init__(self, message: str, path: str = "$", source: str | Path | None = None):
        where = f"{source}: " if source else ""
        super().__init__(f"{where}{path}: {message}")
        self.path = path


class TemplateWarning(UserWarning):
    """Non-fatal template issue (e.g. list items with different structures)."""


# --------------------------------------------------------------------------------------------
# Intermediate representation
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TextLeaf:
    hint: str


@dataclass(frozen=True)
class TypedLeaf:
    kind: str
    hint: str = ""
    nullable: bool = True
    values: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Section:
    fields: dict[str, Node]
    hint: str = ""
    nullable: bool = False


@dataclass(frozen=True)
class ListOf:
    item: Node


Node = Union[TextLeaf, TypedLeaf, Section, ListOf]  # noqa: UP007 (runtime alias)


def _shape(node: Node) -> Any:
    """Structural signature used to warn about heterogeneous list items."""
    if isinstance(node, Section):
        return ("section", tuple((k, _shape(v)) for k, v in node.fields.items()))
    if isinstance(node, ListOf):
        return ("list", _shape(node.item))
    if isinstance(node, TypedLeaf):
        return ("typed", node.kind, node.values)
    return "text"


def _parse_typed_leaf(raw: dict, path: str, source, in_list: bool) -> TypedLeaf:
    unknown = set(raw) - TYPED_LEAF_KEYS
    if unknown:
        raise TemplateError(
            f"unknown key(s) {sorted(map(str, unknown))} in typed leaf; allowed: {sorted(TYPED_LEAF_KEYS)}",
            path,
            source,
        )
    kind = raw["_type"]
    if kind not in TYPED_KINDS:
        raise TemplateError(f"_type must be one of {list(TYPED_KINDS)}, got {kind!r}", path, source)
    hint = raw.get("_hint", "")
    if not isinstance(hint, str):
        raise TemplateError("_hint must be a string", path, source)
    nullable = raw.get("_nullable", not in_list)
    if not isinstance(nullable, bool):
        raise TemplateError("_nullable must be true or false", path, source)
    values = None
    if kind == "enum":
        values = raw.get("values")
        if not isinstance(values, list) or not values:
            raise TemplateError("enum needs a non-empty list 'values'", path, source)
        bad = [v for v in values if not isinstance(v, str)]
        if bad:
            raise TemplateError(
                f"enum values must be strings, got {bad!r} (quote yes/no/on/off/null/numbers in YAML)",
                path,
                source,
            )
        dupes = sorted({v for v in values if values.count(v) > 1})
        if dupes:
            raise TemplateError(f"duplicate enum values {dupes}", path, source)
        values = tuple(values)
    elif "values" in raw:
        raise TemplateError("'values' is only allowed for _type: enum", path, source)
    return TypedLeaf(kind=kind, hint=hint, nullable=nullable, values=values)


def parse_instructions(raw: Any, path: str = "$", source: str | Path | None = None, in_list: bool = False) -> Node:
    """Validate raw YAML data and convert it to the intermediate representation."""
    if isinstance(raw, dict):
        for key in raw:
            if not isinstance(key, str):
                raise TemplateError(
                    f"keys must be strings, got {key!r} (YAML turns yes/no/on/off/numbers into non-strings)",
                    path,
                    source,
                )
        if "_type" in raw:
            if path == "$":
                raise TemplateError("the template root must be a section, not a typed leaf", path, source)
            return _parse_typed_leaf(raw, path, source, in_list)
        reserved = {k for k in raw if k.startswith("_")}
        allowed = SECTION_META_KEYS | (ROOT_ONLY_KEYS if path == "$" else set())
        unknown = reserved - allowed
        if unknown:
            raise TemplateError(
                f"unknown reserved key(s) {sorted(unknown)}; allowed here: {sorted(allowed)} "
                f"(a typed leaf needs _type)",
                path,
                source,
            )
        hint = raw.get("_hint", "")
        nullable = raw.get("_nullable", False)
        if not isinstance(hint, str):
            raise TemplateError("_hint must be a string", path, source)
        if not isinstance(nullable, bool):
            raise TemplateError("_nullable must be true or false", path, source)
        fields = {
            k: parse_instructions(v, f"{path}.{k}", source) for k, v in raw.items() if not k.startswith("_")
        }
        if not fields:
            raise TemplateError("empty section (no fields)", path, source)
        return Section(fields=fields, hint=hint, nullable=nullable)
    if isinstance(raw, list):
        if not raw:
            return ListOf(TextLeaf(""))  # RadFields: [] -> [""]
        items = [parse_instructions(x, f"{path}[]", source, in_list=True) for x in raw]
        first = items[0]
        if any(_shape(x) != _shape(first) for x in items[1:]):
            warnings.warn(
                f"{source or ''}:{path}: list items differ; the first item defines the element type",
                TemplateWarning,
                stacklevel=2,
            )
        return ListOf(first)
    if raw is None:
        return TextLeaf("")
    if isinstance(raw, str):
        return TextLeaf(raw)
    return TextLeaf(str(raw))  # numbers / booleans used as hints (RadFields: str(value))


# --------------------------------------------------------------------------------------------
# Pydantic model and JSON schema
# --------------------------------------------------------------------------------------------


def to_class_name(name: str) -> str:
    """UpperCamelCase class name (RadFields-compatible transliteration)."""
    words = re.split(r"[^0-9A-Za-z]+", _ascii(name))
    out = "".join(w[:1].upper() + w[1:] for w in words if w)
    return out if out and not out[0].isdigit() else f"T{out}"


def _ascii(text: str) -> str:
    text = "".join(_TRANSLIT.get(c, c) for c in text)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def to_field_name(key: str) -> str:
    """Python attribute name for a YAML key; the original key stays the JSON name (alias)."""
    name = re.sub(r"[^0-9A-Za-z_]", "_", _ascii(key)) or "field"
    if name[0].isdigit() or keyword.iskeyword(name) or name in _BASEMODEL_ATTRS or name.startswith("model_"):
        name = f"f_{name}"
    return name.lstrip("_") or "field"


_MODEL_CONFIG = ConfigDict(extra="forbid", populate_by_name=True)


def _annotation(node: Node, class_path: str) -> Any:
    if isinstance(node, TextLeaf):
        return str
    if isinstance(node, TypedLeaf):
        base: Any = {
            "number": float,
            "integer": int,
            "boolean": bool,
            "string": str,
        }.get(node.kind)
        if node.kind == "enum":
            base = Literal[node.values]  # type: ignore[valid-type]
        return Optional[base] if node.nullable else base  # noqa: UP007
    if isinstance(node, ListOf):
        return list[_annotation(node.item, f"{class_path}_Item")]
    model = _section_model(node, class_path)
    return Optional[model] if node.nullable else model  # noqa: UP007


def _section_model(section: Section, class_name: str) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    seen: dict[str, str] = {}
    for key, child in section.fields.items():
        fname = to_field_name(key)
        if fname in seen:
            raise TemplateError(f"keys {seen[fname]!r} and {key!r} collide as field name {fname!r}")
        seen[fname] = key
        ann = _annotation(child, f"{class_name}_{to_class_name(key)}")
        fields[fname] = (ann, Field(..., alias=key))
    return create_model(class_name, __config__=_MODEL_CONFIG, **fields)


def build_model(root: Section, name: str = "Report") -> type[BaseModel]:
    """IR -> Pydantic v2 model. Every field is required; nullable leaves accept null."""
    return _section_model(root, to_class_name(name))


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"].split("/")[-1]
            return _inline_refs(defs[ref], defs)
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(x, defs) for x in node]
    return node


def _clean(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: _clean(v) for k, v in node.items() if k not in {"title", "default", "description", "$defs"}}
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        # Pydantic emits Literal[...] as {"enum": [...], "type": "string"}; keep as is.
        return out
    if isinstance(node, list):
        return [_clean(x) for x in node]
    return node


def vllm_json_schema(model: type[BaseModel]) -> dict:
    """Self-contained JSON schema for vLLM: refs inlined, every property required, no extras."""
    schema = model.model_json_schema(by_alias=True, mode="validation")
    defs = schema.get("$defs", {})
    return _clean(_inline_refs(schema, defs))


def check_schema_portability(schema: Any, path: str = "$") -> list[str]:
    """List keywords outside PORTABLE_KEYWORDS (they may be unsupported by the grammar backend)."""
    problems: list[str] = []
    if isinstance(schema, dict):
        for k, v in schema.items():
            if k not in PORTABLE_KEYWORDS:
                problems.append(f"{path}: keyword {k!r}")
            if k == "properties":
                for name, sub in v.items():
                    problems += check_schema_portability(sub, f"{path}.{name}")
            elif k == "items":
                problems += check_schema_portability(v, f"{path}[]")
            elif k == "anyOf":
                for i, sub in enumerate(v):
                    problems += check_schema_portability(sub, f"{path}|{i}")
    return problems


# --------------------------------------------------------------------------------------------
# Reader view (the template as shown to the LLM)
# --------------------------------------------------------------------------------------------


def _leaf_repr(node: TypedLeaf | TextLeaf) -> str:
    if isinstance(node, TextLeaf):
        return json.dumps(node.hint, ensure_ascii=False)
    if node.kind == "enum":
        rep = " | ".join(json.dumps(v, ensure_ascii=False) for v in node.values or ())
    else:
        rep = {"number": "<number>", "integer": "<integer>", "boolean": "true | false", "string": '"<text>"'}[
            node.kind
        ]
    return f"{rep} | null" if node.nullable else rep


def render_reader_view(root: Section) -> str:
    """Annotated JSON skeleton: allowed values, null, and hints as // comments."""
    lines: list[str] = []

    def emit(node: Node, indent: int, prefix: str, comma: str) -> None:
        pad = "  " * indent
        if isinstance(node, Section):
            notes = [n for n in (node.hint, "object or null" if node.nullable else "") if n]
            comment = f"  // {'; '.join(notes)}" if notes else ""
            lines.append(f"{pad}{prefix}{{{comment}")
            items = list(node.fields.items())
            for i, (key, child) in enumerate(items):
                emit(child, indent + 1, f"{json.dumps(key, ensure_ascii=False)}: ", "," if i < len(items) - 1 else "")
            lines.append(f"{pad}}}{comma}")
        elif isinstance(node, ListOf):
            item = node.item
            if isinstance(item, (TextLeaf, TypedLeaf)):
                comment = f"  // list, may be empty{'; ' + item.hint if item.hint and isinstance(item, TypedLeaf) else ''}"
                lines.append(f"{pad}{prefix}[{_leaf_repr(item)}, ...]{comma}{comment}")
            else:
                lines.append(f"{pad}{prefix}[  // list, one object per entry, may be empty")
                emit(item, indent + 1, "", ", ...")
                lines.append(f"{pad}]{comma}")
        else:
            hint = node.hint if isinstance(node, TypedLeaf) else ""
            comment = f"  // {hint}" if hint else ""
            lines.append(f"{pad}{prefix}{_leaf_repr(node)}{comma}{comment}")

    emit(root, 0, "", "")
    return "\n".join(lines)


def render_reader_view_radfields(raw: Any) -> str:
    """RadFields-style reader view: json.dumps of the instructions (hints as values)."""

    def conv(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items() if not k.startswith("_")}
        if isinstance(x, list):
            return [conv(i) for i in x] or [""]
        return "" if x is None else str(x)

    return json.dumps(conv(raw), ensure_ascii=False)


def field_catalog(root: Section) -> list[dict[str, Any]]:
    """Flat list of leaves: path, kind, values, nullable, hint (documentation, export, QA generators)."""
    out: list[dict[str, Any]] = []

    def walk(node: Node, path: str) -> None:
        if isinstance(node, Section):
            for k, v in node.fields.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, ListOf):
            walk(node.item, f"{path}[]")
        elif isinstance(node, TypedLeaf):
            out.append(
                {"path": path, "kind": node.kind, "values": list(node.values or []), "nullable": node.nullable,
                 "hint": node.hint}
            )
        else:
            out.append({"path": path, "kind": "text", "values": [], "nullable": False, "hint": node.hint})

    walk(root, "")
    return out


# --------------------------------------------------------------------------------------------
# Template bundle (folder) and prompts
# --------------------------------------------------------------------------------------------


class TemplateConfig(BaseModel):
    """templates/<name>/config.yaml (every key optional)."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    missing_value: str = ""  # written by the LLM into text fields without information
    reader_view_style: Literal["annotated", "radfields"] = "annotated"
    feeder: str | None = None  # default feeder (--feeder overrides)
    feeder_options: dict[str, Any] = Field(default_factory=dict)
    request: dict[str, Any] = Field(default_factory=dict)  # overrides of the model's request defaults
    evidence_fields: list[str] = Field(default_factory=lambda: ["evidence"])  # key names, any depth
    export_tables: dict[str, str] = Field(default_factory=dict)  # table name -> top-level list key
    postprocess: str | None = None  # python file in the template folder (see export.py)


@dataclass
class Template:
    name: str
    folder: Path | None  # None for a bare YAML file
    files: dict[str, Path]  # role -> file actually used (template, system_prompt, user_prompt, ...)
    raw: dict
    root: Section
    config: TemplateConfig
    model: type[BaseModel]
    json_schema: dict
    reader_view: str
    system_prompt: str
    user_prompt: str
    hash: str = ""
    warnings: list[str] = field(default_factory=list)

    def render_messages(self, report_text: str) -> list[dict[str, str]]:
        """System message identical for every report (prefix caching); the report goes in the user message."""
        user = self.user_prompt.replace("{{report}}", report_text)
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]

    def postprocess_path(self) -> Path | None:
        return self.files.get("postprocess")

    def snapshot(self, dest: Path) -> None:
        """Copy the files used by this template into dest (kept with the run outputs)."""
        dest.mkdir(parents=True, exist_ok=True)
        for role, path in self.files.items():
            shutil.copy2(path, dest / _SNAPSHOT_NAMES.get(role, path.name))
        (dest / "schema.json").write_text(json.dumps(self.json_schema, indent=1))
        (dest / "rendered_system_prompt.md").write_text(self.system_prompt)


_SNAPSHOT_NAMES = {
    "template": "template.yaml",
    "system_prompt": "system_prompt.md",
    "user_prompt": "user_prompt.md",
    "config": "config.yaml",
    "postprocess": "postprocess.py",
}


def resolve_template(ref: str | Path, templates_dir: Path) -> Path:
    """A name under templates/, a template folder, or a single YAML file (pure RadFields template)."""
    p = Path(ref).expanduser()
    if p.exists():
        return p.resolve()
    candidate = templates_dir / str(ref)
    if candidate.is_dir():
        return candidate.resolve()
    known = sorted(d.name for d in templates_dir.iterdir() if d.is_dir() and not d.name.startswith("_"))
    raise FileNotFoundError(f"template {ref!r} not found (neither a path nor templates/{ref}); known: {known}")


def _fill(text: str, values: dict[str, str]) -> str:
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    return text


def load_template(ref: str | Path, templates_dir: Path | None = None) -> Template:
    """Load a template folder (template.yaml [+ system_prompt.md, user_prompt.md, config.yaml,
    postprocess.py]) or a single YAML file. Missing prompts fall back to templates/_default/."""
    templates_dir = templates_dir or REPO_ROOT / "templates"
    path = resolve_template(ref, templates_dir)
    files: dict[str, Path] = {}
    if path.is_dir():
        folder: Path | None = path
        name = path.name
        files["template"] = path / "template.yaml"
        if not files["template"].is_file():
            raise TemplateError("template folder has no template.yaml", source=path)
    else:
        folder = None
        name = path.stem
        files["template"] = path
    for role in ("system_prompt", "user_prompt"):
        own = folder / f"{role}.md" if folder else None
        files[role] = own if own and own.is_file() else DEFAULT_PROMPTS_DIR / f"{role}.md"
    config = TemplateConfig()
    if folder and (folder / "config.yaml").is_file():
        files["config"] = folder / "config.yaml"
        try:
            config = TemplateConfig.model_validate(yaml.safe_load(files["config"].read_text()) or {})
        except Exception as e:  # pydantic.ValidationError -> readable message
            raise TemplateError(f"invalid config.yaml: {e}", source=files["config"]) from e
    if config.postprocess:
        if folder is None:
            raise TemplateError("postprocess needs a template folder")
        files["postprocess"] = folder / config.postprocess
        if not files["postprocess"].is_file():
            raise TemplateError(f"postprocess file {config.postprocess} not found", source=folder)

    try:
        raw = yaml.safe_load(files["template"].read_text())
    except yaml.YAMLError as e:
        raise TemplateError(f"YAML error: {e}", source=files["template"]) from e
    if not isinstance(raw, dict):
        raise TemplateError("the template root must be a mapping", source=files["template"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", TemplateWarning)
        root = parse_instructions(raw, source=files["template"])
    assert isinstance(root, Section)
    try:
        model = build_model(root, name)
    except TemplateError as e:
        raise TemplateError(str(e), source=files["template"]) from e
    schema = vllm_json_schema(model)
    problems = check_schema_portability(schema)
    if problems:
        raise TemplateError(f"schema uses non-portable keywords: {problems[:10]}", source=files["template"])

    reader_view = render_reader_view(root) if config.reader_view_style == "annotated" else render_reader_view_radfields(raw)
    values = {"reader_view": reader_view, "missing_value": config.missing_value}
    system_prompt = _fill(files["system_prompt"].read_text(), values).strip() + "\n"
    user_prompt = _fill(files["user_prompt"].read_text(), values).strip() + "\n"
    if "{{report}}" not in user_prompt:
        raise TemplateError("user prompt must contain the {{report}} placeholder", source=files["user_prompt"])

    tmpl = Template(
        name=name,
        folder=folder,
        files=files,
        raw=raw,
        root=root,
        config=config,
        model=model,
        json_schema=schema,
        reader_view=reader_view,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        warnings=[str(w.message) for w in caught],
    )
    tmpl.hash = template_hash(tmpl)
    return tmpl


def template_hash(t: Template) -> str:
    """sha256 of what is sent to the model (prompts, schema) and of the config.

    Stored in every output record: a run refuses to resume with a different template, so one
    results file never mixes extractions made with two template versions.
    """
    payload = {
        "system": t.system_prompt,
        "user": t.user_prompt,
        "schema": t.json_schema,
        "config": t.config.model_dump(),
    }
    pp = t.postprocess_path()
    if pp:
        payload["postprocess"] = pp.read_text()
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
