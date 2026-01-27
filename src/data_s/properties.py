from src.data_s.mcc_tools import load_json, summarize_file_obj
from src.preprocessing.comment_process import strip_cryptol_comments_all
from typing import Dict
import re

_MODULE_HEADER_RE = re.compile(r"(?m)^\s*module\s+\S+(\s*=\s*\S+)?\s+where\s*$")

def get_simple_instruction_limits(
    num_declarations: int,
    total_mcc: float
) -> Dict[str, int | str | bool]:
    is_wrapper = num_declarations <= 2 and total_mcc <= 1.0
    if is_wrapper:
        return {
            "instruction_max_words": 28,
            "instruction_detail_level": "minimal",
            "is_thin_wrapper": True,
        }
    
    budget = 20
    budget += min(60, num_declarations * 4)
    budget += min(70, int(total_mcc * 6))

    budget = max(25, min(150, budget))

    return {
        "instruction_max_words": int(budget),
        "instruction_detail_level": "standard" if budget < 110 else "detailed",
        "is_thin_wrapper": False,
    }

def contains_module(code_excerpt: str) -> bool:
    # Check if the word 'module' appears in the code excerpt
    code_excerpt = strip_cryptol_comments_all(code_excerpt)
    return bool(
        _MODULE_HEADER_RE.search(code_excerpt)
        )

from typing import Dict, Optional

def get_masked_definition_instruction_limits(
    *,
    total_mcc: float,
    num_declarations: int,
    max_mcc: Optional[float] = None,
    num_params: int = 0,
    kind: str = "declaration",   # "property" or "declaration"
) -> Dict[str, int | str | bool]:
    """
    Instruction limits for *fill-the-hole* examples.

    Goal: very short, because the model sees the masked source as input.
    """

    # Normalize inputs
    total_mcc = float(total_mcc or 0.0)
    num_declarations = int(num_declarations or 0)
    mmax = float(max_mcc) if max_mcc is not None else None

    # Baseline: one-line instruction
    budget = 14

    # Complexity: use max_mcc if available (better signal than total_mcc for localized difficulty)
    if mmax is not None:
        budget += min(18, int(max(0.0, mmax - 1.0) * 3))   # +0..18
    else:
        budget += min(12, int(max(0.0, total_mcc) * 0.7))  # weaker fallback

    # Larger files slightly increase the needed specificity (but keep it small)
    budget += min(8, num_declarations // 6)               # +0..8

    # Params add a tiny bit of needed clarity
    budget += min(6, num_params * 2)                      # +0..6

    # Properties are often “one statement”, keep them extra short
    if kind == "property":
        budget -= 3

    # Clamp hard
    budget = max(10, min(45, budget))

    return {
        "instruction_max_words": int(budget),
        "instruction_detail_level": "minimal" if budget <= 22 else "standard",
        "is_thin_wrapper": True if budget <= 28 else False,
  
    }

def get_masked_definition_instruction_limits(
    *,
    total_mcc: float,
    num_declarations: int,
    max_mcc: Optional[float] = None,
    num_params: int = 0,
    kind: str = "declaration",
) -> Dict[str, int | str | bool]:
    total_mcc = float(total_mcc or 0.0)
    num_declarations = int(num_declarations or 0)
    mmax = float(max_mcc) if max_mcc is not None else None

    budget = 14
    if mmax is not None:
        budget += min(18, int(max(0.0, mmax - 1.0) * 3))
    else:
        budget += min(12, int(max(0.0, total_mcc) * 0.7))

    budget += min(8, num_declarations // 6)
    budget += min(6, num_params * 2)

    if kind == "property":
        budget -= 3

    budget = max(10, min(45, budget))

    # sentence guidance
    force_one = budget <= 28
    max_sentences = 1 if force_one else 2

    return {
        "instruction_max_words": int(budget),
        "is_thin_wrapper": bool(force_one),
        "instruction_detail_level": "minimal" if budget <= 22 else "standard",
    }
