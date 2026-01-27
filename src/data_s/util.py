from __future__ import annotations

from pathlib import Path
import pandas as pd
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from src.data_s.mcc_tools import load_json, get_hardest_definition


_TOP_LEVEL_START_RE = re.compile(
    r"^(module|import|type|property|parameter|interface|private)\b"
    r"|^[A-Za-z_][A-Za-z0-9_']*\b"
    r"|\([^)]*\)\s*[:=]"   # operator signature/definition
)


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))

def _is_comment_or_blank(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("//")

def _is_block_comment_start(line: str) -> bool:
    return line.lstrip().startswith("/*")

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


def _starts_property_definition(line: str, name: str) -> bool:
    return re.match(rf"^\s*property\s+{re.escape(name)}\b.*?\s*=\s*", line) is not None

def _starts_any_definition(line: str, name: str) -> bool:
    return _starts_definition(line, name) or _starts_property_definition(line, name)

def _find_definition_span(lines: List[str], name: str) -> Optional[Tuple[int, int]]:
    start = None
    for i, line in enumerate(lines):
        if _starts_any_definition(line, name):
            start = i
            break
    if start is None:
        return None

    base_indent = _indent_of(lines[start])

    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if _is_blank_or_comment(ln) or _is_block_comment_start(ln):
            continue

        # indentation-based stop
        if _indent_of(ln) < base_indent:
            end = j
            break

        # also stop on a new top-level start when base_indent==0
        if base_indent == 0 and _is_top_level_start(ln):
            # allow multiple equations for same name
            if re.match(rf"^{re.escape(name)}\b", ln) or re.match(rf"^\s*property\s+{re.escape(name)}\b", ln):
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
    kind_prefix: Optional[str] = None,   # NEW
) -> List[str]:
    out: List[str] = []
    if docstring:
        out.append(f"/** {docstring} */")

    args = (" " + " ".join(params)) if params else ""
    prefix = (kind_prefix + " ") if kind_prefix else ""
    out.append(f"{prefix}{name}{args} = {hole_name}\n")
    return out



@dataclass(frozen=True)
class MaskedResult:
    masked_source: str
    removed_definition: str
    hole_name: str

def _normalize_inline_sig_and_def(lines: List[str], name: str) -> List[str]:
    """
    If a line looks like: 'name : ... name ... = ...'
    split into two lines:
      1) 'name : ...'
      2) 'name ... = ...'
    """
    out: List[str] = []
    # signature then later another 'name' token
    pat = re.compile(rf"^(\s*{re.escape(name)}\s*:\s*.*?)(\s+{re.escape(name)}\b.*)$")
    for line in lines:
        m = pat.match(line)
        if m:
            out.append(m.group(1).rstrip())
            out.append(m.group(2).lstrip())
        else:
            out.append(line)
    return out


def _is_property_line(line: str, name: str) -> bool:
    return re.match(rf"^\s*property\s+{re.escape(name)}\b", line) is not None


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
    lines = _normalize_inline_sig_and_def(lines, name)
    span = _find_definition_span(lines, name)

    if span is None:
        # Debug: show all lines that contain the name, with repr() to reveal hidden chars.
        hits = [(i, repr(l)) for i, l in enumerate(lines) if name in l]
        print("DEBUG name hits:", hits[:20])
        # Also show lines where your predicate *almost* matches:
        near = []
        for i, l in enumerate(lines):
            if l.lstrip().startswith(name):
                near.append((i, repr(l)))
        print("DEBUG lstrip startswith hits:", near[:20])
        raise ValueError(f"Could not find definition span for '{name}'")

    start, end = span
    removed = "\n".join(lines[start:end])

    kind_prefix = "property" if _is_property_line(lines[start], name) else None

    placeholder = make_placeholder_block(
        name=name,
        params=params,
        hole_name=hole_name,
        docstring=docstring,
        kind_prefix=kind_prefix,  
    )

    new_lines = lines[:start] + placeholder + lines[end:]
    return MaskedResult(
        masked_source="\n".join(new_lines),
        removed_definition=removed,
        hole_name=hole_name,
    )

def build_masked_examples_df(source_code_df, hole_name="/* Finish this definition. */"):
    rows = []
    for _, row in source_code_df.iterrows():
        #print(f"{_}: Processing {row['filename']}")
        mcc_obj = load_json(row["json_path"])
        definition = get_hardest_definition(mcc_obj)

        if not definition:
            continue
        try:
            masked = mask_declaration_in_source(
                source=row["content"],
                name=definition["name"],
                params=definition.get("params", []),
                hole_name=hole_name,
        )

        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")
            continue
        rows.append({
            **row.to_dict(),
            "def_name": definition["name"],
            "def_params": definition.get("params", []),
            "masked_source": masked.masked_source,
            "target_definition": masked.removed_definition,
            "hole_name": masked.hole_name,

        })
    return pd.DataFrame(rows)
