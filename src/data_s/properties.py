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