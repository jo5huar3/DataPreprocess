from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd

def make_messages_row(
    row: pd.Series,
    system_prompt: str,
    include_input: bool = True,
) -> List[Dict[str, str]]:
    """
    Convert one dataframe row into OpenAI-style messages format.
    Expects: instruction, input (optional), output.
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    user_parts = [str(row.get("instruction", "")).strip()]
    if include_input:
        inp = str(row.get("input", "")).strip()
        if inp:
            user_parts.append("\n\nInput:\n" + inp)

    user_content = "\n".join([p for p in user_parts if p])

    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": str(row.get("output", "")).strip()})
    return messages

def add_messages_column(
    df: pd.DataFrame,
    system_prompt: str,
    out_col: str = "messages",
    include_input: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    df[out_col] = df.apply(make_messages_row, axis=1, system_prompt=system_prompt, include_input=include_input)
    return df
