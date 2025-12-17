#!/usr/bin/env python3
"""
Parse all .cry (Cryptol) and .saw (SAW) files under one or more directories (recursively)
and export to JSONL.

Each line in the JSONL has:
  - filename: absolute file path (or stripped path if --strip is provided)
  - filetype: 'cry' or 'saw'
  - content: file contents as UTF-8 (with replacement for invalid bytes)

Usage:
  python code_parser.py /path/to/repo [/another/root ...] --out data/sources.jsonl --strip /path/to

If --out is omitted, defaults to ./data/cryptol_sources.jsonl
"""
from __future__ import annotations
import sys
import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

# --- Helpers -----------------------------------------------------------------

def iter_source_files(root: Path, exts: Iterable[str]=(".cry", ".saw")) -> Iterator[Path]:
    """Yield Path objects for files under `root` matching any extension in `exts` (case-insensitive)."""
    exts_lower = {e.lower() for e in exts}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_lower:
            yield p

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def read_text_safe(p: Path) -> str:
    # Read as UTF-8, replacing undecodable bytes so we never crash
    return p.read_text(encoding="utf-8", errors="replace")

def strip_prefix(full: Path, prefix: Optional[Path]) -> str:
    """
    If `prefix` is provided and is a real prefix of `full`, return the relative string path.
    Otherwise return the absolute path. Always uses POSIX separators for consistency.
    """
    full = full.resolve()
    if prefix is None:
        return full.as_posix()
    try:
        rel = full.relative_to(prefix.resolve())
        return rel.as_posix()
    except ValueError:
        # Not a prefix; fall back to absolute
        return full.as_posix()

# --- Optional utility ---------------------------------------------------------

def jsonl_to_dataframe(absolute_path: str):
    """
    Load a JSONL file into a pandas DataFrame. (Optional utility.)
    Expects per-line dicts with at least: filename, relpath, filetype, content, root.
    """
    import pandas as pd
    records = []
    with open(absolute_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame.from_records(records)

# --- Main --------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect Cryptol (.cry) and SAW (.saw) files into a JSONL dataset."
    )
    parser.add_argument(
        "roots",
        nargs="+",
        help="One or more root directories to search."
    )
    parser.add_argument(
        "--out", "-o",
        default="data/cryptol_sources.jsonl",
        help="Output JSONL path (default: data/cryptol_sources.jsonl)."
    )
    parser.add_argument(
        "--strip",
        metavar="PATH",
        type=str,
        default=None,
        help="If provided, strip this leading path prefix from 'filename' (nice for repo-relative paths)."
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the JSONL instead of overwriting."
    )

    parser.add_argument(
        "--ext", "--include-ext",
        nargs="+",
        default=[".cry", ".saw"],
        help="File extensions to include (space-separated). Example: --ext .cry .saw .md"
    )

    args = parser.parse_args(argv)

    # Convert ext to .ext format
    exts = []
    for e in args.ext:
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        exts.append(e)

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    out_path = Path(args.out).expanduser().resolve()
    strip_path = Path(args.strip).expanduser().resolve() if args.strip else None

    ensure_parent(out_path)

    mode = "a" if args.append and out_path.exists() else "w"
    total = 0
    errors: list[str] = []

    with open(out_path, mode, encoding="utf-8") as out_f:
        for root in roots:
            root = root.resolve()
            for p in iter_source_files(root, exts=exts):
                try:
                    content = read_text_safe(p)
                except Exception as e:
                    print(f"[WARN] Could not read {p}: {e}", file=sys.stderr)
                    errors.append(str(p))
                    continue

                # filetype from suffix
                suff = p.suffix.lower()
                if suff not in (".cry", ".saw"):
                    # Defensive: should not happen due to filter above
                    continue
                filetype = suff.lstrip(".")

                # filename with optional strip, relpath always relative to current root
                filename = strip_prefix(p, strip_path)
                record = {
                    "filename": filename,
                    "filetype": filetype,
                    "content": content,
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

    print(f"Wrote {total} files (.cry/.saw) to {out_path}")
    if errors:
        print(f"[INFO] {len(errors)} files could not be read. First few:", errors[:5])

if __name__ == "__main__":
    main()
