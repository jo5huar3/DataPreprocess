from __future__ import annotations

from pathlib import Path
import pandas as pd
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


_TOP_LEVEL_START_RE = re.compile(
    r"^(module|import|type|property)\b|^[A-Za-z_][A-Za-z0-9_']*\b"
)


def jsonl_append_dedup(
    df: pd.DataFrame,
    path: str | Path,
    key_cols: Tuple[str, str] = ("filename", "scenario"),
    overwrite_existing: bool = False,
) -> Path:
    """
    Append df to a JSONL file (records/lines) while avoiding duplicates based on key_cols.

    - overwrite_existing=False (default):
        Keep what's already in the file; append only new (filename, scenario) keys.
    - overwrite_existing=True:
        Replace any existing records that share a key with incoming df rows.
        (i.e., upsert by key: new rows win)

    Notes:
      - JSONL is row-oriented; "overwrite" is implemented by rewriting the full file
        when overwrite_existing=True.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate key columns exist
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing key columns: {missing}")

    df_in = df.copy()

    # If file doesn't exist or empty, just write
    if (not path.exists()) or path.stat().st_size == 0:
        df_in.to_json(path, orient="records", lines=True, mode="w", force_ascii=False)
        return path

    # Load existing file
    existing_df = pd.read_json(path, lines=True)

    # Ensure existing has key cols (otherwise we can't dedupe/overwrite safely)
    missing_existing = [c for c in key_cols if c not in existing_df.columns]
    if missing_existing:
        raise ValueError(
            f"Existing JSONL at {path} is missing key columns: {missing_existing}. "
            "Cannot dedupe/overwrite by key."
        )

    if overwrite_existing:
        # Upsert: drop existing rows that match incoming keys, then append incoming, then rewrite file
        incoming_keys = set(map(tuple, df_in[list(key_cols)].itertuples(index=False, name=None)))
        existing_keys = list(map(tuple, existing_df[list(key_cols)].itertuples(index=False, name=None)))

        keep_mask = [k not in incoming_keys for k in existing_keys]
        merged = pd.concat([existing_df.loc[keep_mask], df_in], ignore_index=True)

        merged.to_json(path, orient="records", lines=True, mode="w", force_ascii=False)
        return path

    # Default: append only truly new keys
    existing_keys = set(map(tuple, existing_df[list(key_cols)].itertuples(index=False, name=None)))
    new_keys = list(map(tuple, df_in[list(key_cols)].itertuples(index=False, name=None)))
    keep_mask = [k not in existing_keys for k in new_keys]

    df_to_add = df_in.loc[keep_mask]
    if df_to_add.empty:
        return path

    df_to_add.to_json(path, orient="records", lines=True, mode="a", force_ascii=False)
    return path




def _is_blank_or_comment(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("//")


def _is_top_level_start(line: str) -> bool:
    # Require column 0 to avoid grabbing indented where/let content.
    if not line or line[:1].isspace():
        return False
    if _is_blank_or_comment(line):
        return False
    if line.lstrip().startswith("/*"):
        return False
    return _TOP_LEVEL_START_RE.match(line) is not None


def _starts_signature(line: str, name: str) -> bool:
    return re.match(rf"^\s*{re.escape(name)}\s*:\s*", line) is not None


def _starts_definition(line: str, name: str) -> bool:
    """
    Match 'name ... = ...' but avoid matching type constraints '=>'
    by excluding signature lines and requiring an assignment '='.
    """
    # Exclude signature line form "name : ..."
    if _starts_signature(line, name):
        return False
    return re.match(rf"^\s*{re.escape(name)}\b(?!\s*:).*?\s=\s*", line) is not None


def _find_definition_span(lines: List[str], name: str) -> Optional[Tuple[int, int]]:
    start = None
    for i, line in enumerate(lines):
        if _starts_definition(line, name):
            start = i
            break
    if start is None:
        return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _is_top_level_start(lines[j]):
            # allow multiple equations starting with same name
            if re.match(rf"^{re.escape(name)}\b", lines[j]):
                continue
            end = j
            break
    return start, end


def make_placeholder_block(
    *,
    name: str,
    params: List[str],
    hole_name: str,
    docstring: Optional[str] = None,
) -> List[str]:
    """
    HumanEval-ish: put a docstring comment immediately above the stub.
    Cryptol supports // and /* */ and docstrings /** ... */. :contentReference[oaicite:2]{index=2}
    """
    out: List[str] = []
    if docstring:
        out.append(f"/** {docstring} */")
    args = (" " + " ".join(params)) if params else ""
    out.append(f"{name}{args} = {hole_name}\n")
    return out


@dataclass(frozen=True)
class MaskedResult:
    masked_source: str
    removed_definition: str
    hole_name: str


def mask_declaration_in_source(
    source: str,
    *,
    name: str,
    params: List[str],
    hole_name: Optional[str] = None,
    docstring: Optional[str] = None,
) -> MaskedResult:
    """
    Remove the *definition equations* for `name` and replace with a stub:
      /** docstring */
      name <params> = __HOLE_name

    Leaves any existing signature line in place.
    """
    hole_name = hole_name or f"__HOLE_{name}"
    lines = source.splitlines()
    span = _find_definition_span(lines, name)
    if span is None:
        raise ValueError(f"Could not find definition span for '{name}'")

    start, end = span
    removed = "\n".join(lines[start:end])

    placeholder = make_placeholder_block(
        name=name,
        params=params,
        hole_name=hole_name,
        docstring=docstring,
    )

    new_lines = lines[:start] + placeholder + lines[end:]
    return MaskedResult(
        masked_source="\n".join(new_lines),
        removed_definition=removed,
        hole_name=hole_name,
    )
