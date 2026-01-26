from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd


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