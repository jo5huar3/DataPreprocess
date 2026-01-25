#!/usr/bin/env python3
"""
make_alpaca_instructions_pydantic_ai.py

Refactor of your OpenAI Responses `responses.parse(...)` based generator to:
  ✅ use pydantic_ai (Pydantic AI Agents)
  ✅ move prompts to Jinja templates
  ✅ inject local RAG context from your rag.py (ChromaDB + OpenAI embeddings)

This module focuses on the "LLM call" portion of your pipeline. It is designed
to be a drop-in replacement for:
  - call_openai_structured(...)
  - build_prompt_call_openai_structured(...)

Assumptions:
- You have Jinja templates on disk (see ./prompts/*.j2).
- You have your `rag.py` available in your import path (same package, or top-level).
- You set OPENAI_API_KEY in your environment for both pydantic_ai + embeddings.

Dependencies:
  pip install pydantic-ai jinja2 openai chromadb python-dotenv

Docs:
- Pydantic AI overview / Agents / OpenAI provider:
  https://ai.pydantic.dev/
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
import dotenv
import pandas as pd
from pydantic import BaseModel

# ---- Your existing cache utility (kept unchanged) ----
from src.util.file_kv_cache import FileKVCache
dotenv.load_dotenv()

# ---------- Import your rag.py (supports both relative + absolute) ----------
try:
    # If rag.py sits next to this module in a package:
    from . import rag  # type: ignore
except Exception:
    import rag  # type: ignore

# ---------- Pydantic AI ----------
from pydantic_ai import Agent

# ---------- Jinja ----------
from jinja2 import Environment, FileSystemLoader, StrictUndefined


# =========================
#   Output Schemas
# =========================
class AlpacaRow(BaseModel):
    instruction: str
    input: str
    output: str


class QAPair(BaseModel):
    question: str
    answer: str


class QAPairList(BaseModel):
    qa_pairs: List[QAPair]


# =========================
#   Config
# =========================
@dataclass(frozen=True)
class RAGConfig:
    enabled: bool = os.getenv("RAG_ENABLED", "1").lower() not in ("0", "false", "no")
    persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "md_kb")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    max_context_chars: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4500"))


@dataclass(frozen=True)
class PromptConfig:
    """
    Where prompts live + knobs for how much source we show.
    """
    template_dir: str = os.getenv("PROMPT_TEMPLATE_DIR", str(Path(__file__).parent / "prompts"))
    # How many chars of code to include in the prompt
    max_code_chars: int = int(os.getenv("PROMPT_MAX_CODE_CHARS", "6000"))
    # How large the "input" field in AlpacaRow is allowed to be
    input_max_chars: int = int(os.getenv("PROMPT_INPUT_MAX_CHARS", "512"))


@dataclass(frozen=True)
class ModelConfig:
    """
    Pydantic AI accepts model identifiers like 'openai:gpt-4.1-mini'.
    """
    model: str = os.getenv("SYNTH_MODEL", "gpt-4.1-mini")
    retries: int = int(os.getenv("SYNTH_RETRIES", "2"))

@dataclass(frozen=True)
class TemplateBundle:
    # “spec generation” (AlpacaRow)
    spec_system: str = "system_spec_instruction.j2"
    spec_user: str = "user_alpaca_instruction.j2"

# =========================
#   Prompt rendering
# =========================
class PromptRenderer:
    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, name: str, **ctx: Any) -> str:
        return self.env.get_template(name).render(**ctx)

class AgentRegistry:
    def __init__(self, model_cfg: ModelConfig, renderer: PromptRenderer):
        self.model_name = _normalize_model(model_cfg.model)
        self.retries = model_cfg.retries
        self.renderer = renderer
        self._cache: Dict[Tuple[str, Type[BaseModel]], Agent[Any, Any]] = {}

    def get(self, system_template: str, output_type: Type[BaseModel]) -> Agent[Any, Any]:
        key = (system_template, output_type)
        if key in self._cache:
            return self._cache[key]

        system_prompt = self.renderer.render(system_template).strip()
        agent = Agent(
            self.model_name,
            output_type=output_type,
            system_prompt=system_prompt,
            retries=self.retries,
        )
        self._cache[key] = agent
        return agent

# =========================
#   RAG helpers
# =========================
_CRYPTOL_MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z0-9_:$]+)", re.MULTILINE)
_CRYPTOL_IMPORT_RE = re.compile(r"^\s*import\s+`?([A-Za-z0-9_:$:./-]+)`?", re.MULTILINE)
_SAW_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_.$:/-]+)", re.MULTILINE)

def _agent_run_compat(agent, user_prompt: str):
    """
    Run a pydantic_ai Agent in BOTH:
      - normal scripts (no running loop)
      - notebooks (loop already running)

    Returns the pydantic_ai RunResult (same shape as run_sync).
    """
    async def _coro():
        return await agent.run(user_prompt)

    try:
        # If this succeeds, we're in an environment with a running loop (e.g., Jupyter)
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if not running:
        # Normal python execution
        return asyncio.run(_coro())

    # Jupyter / already-running loop: run in a separate thread with its own loop
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: asyncio.run(_coro()))
        return fut.result()

def _safe_excerpt(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n…(truncated)…\n"


def build_rag_query(filename: str, lang: str, code: str) -> str:
    """
    Build a retrieval query that tends to work well for Cryptol/SAW code:
    include file path + module name + imports + a short excerpt.
    """
    code = code or ""
    module = None
    imports: List[str] = []

    if lang.lower() == "cryptol":
        m = _CRYPTOL_MODULE_RE.search(code)
        module = m.group(1) if m else None
        imports = list(dict.fromkeys(_CRYPTOL_IMPORT_RE.findall(code)))[:20]
    elif lang.lower() in ("saw", "sawscript"):
        imports = list(dict.fromkeys(_SAW_IMPORT_RE.findall(code)))[:20]

    parts: List[str] = [
        f"language: {lang}",
        f"file: {filename}",
    ]
    if module:
        parts.append(f"module: {module}")
    if imports:
        parts.append("imports: " + ", ".join(imports))
    parts.append("code excerpt:\n" + _safe_excerpt(code, 1200))

    return "\n".join(parts)


def get_rag_context(query: str, rag_cfg: RAGConfig) -> str:
    if not rag_cfg.enabled:
        return ""

    try:
        hits = rag.retrieve(
            query=query,
            persist_dir=rag_cfg.persist_dir,
            collection_name=rag_cfg.collection_name,
            k=rag_cfg.top_k,
            embedding_model=rag_cfg.embedding_model,
        )
        return rag.build_context(hits, max_chars=rag_cfg.max_context_chars)
    except Exception as e:
        # Retrieval should never kill dataset generation.
        # If you want to fail hard, remove this.
        return f"(RAG unavailable: {type(e).__name__}: {e})"


# =========================
#   Pydantic AI Agents
# =========================
def _normalize_model(model: str) -> str:
    """
    Accept either:
      - 'gpt-4.1-mini'  -> 'openai:gpt-4.1-mini'
      - 'openai:gpt-4.1-mini' (unchanged)
      - 'openai-responses:gpt-4.1-mini' (unchanged)
    """
    if ":" in model:
        return model
    return f"openai:{model}"

def build_agents(
    model_cfg: ModelConfig,
    prompt_renderer: PromptRenderer,
    *,
    spec_templates: TemplateBundle,
) -> Dict[str, Agent[Any, Any]]:
    model_name = _normalize_model(model_cfg.model)

    # Spec agent (AlpacaRow)
    system_spec = prompt_renderer.render(spec_templates.spec_system).strip()
    alpaca_agent: Agent[None, AlpacaRow] = Agent(
        model_name,
        output_type=AlpacaRow,
        system_prompt=system_spec,
        retries=model_cfg.retries,
    )

    agent = alpaca_agent


    return agent

# =========================
#   Main call (replacement)
# =========================
def build_prompt_call_pydantic_ai(
    agent: Agent[Any, Any],
    prompt_cfg: PromptConfig,
    rag_cfg: RAGConfig,
    prompt_renderer: PromptRenderer,
    *,
    templates: TemplateBundle,
    input_mode: str,
    filename: str,
    lang: str,
    code: str,
    extra_vars: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    code_excerpt = _safe_excerpt(code, prompt_cfg.max_code_chars)

    rag_context = ""
    if rag_cfg.enabled:
        rag_query = build_rag_query(filename=filename, lang=lang, code=code)
        rag_context = get_rag_context(rag_query, rag_cfg)

    ctx: Dict[str, Any] = {
        "filename": filename,
        "lang": lang,
        "code_excerpt": code_excerpt,
        "rag_context": rag_context,
        "input_max_chars": prompt_cfg.input_max_chars,
    }
    if extra_vars:
        ctx.update(extra_vars)

    user_prompt = prompt_renderer.render(templates.spec_user, **ctx)
    print(user_prompt)
    
    run_result = _agent_run_compat(agent, user_prompt)
    out: AlpacaRow = run_result.output

    out.output = ""
    out.input = "" if input_mode == "none" else _safe_excerpt(code, prompt_cfg.input_max_chars).replace("\n…(truncated)…\n", "")

    # If you NEVER want code fences in instruction:
    if "```" in out.instruction:
        out.instruction = out.instruction.replace("```cryptol", "").replace("```", "").strip()

    return out.model_dump()

def iter_call_pydantic_ai(
    input_df: pd.DataFrame,
    *,
    input_mode: str,
    file_cache_path: str,
    model_cfg: Optional[ModelConfig] = None,
    prompt_cfg: Optional[PromptConfig] = None,
    rag_cfg: Optional[RAGConfig] = None,
    spec_templates: Optional[TemplateBundle] = None,           # NEW
    qa_templates: Optional["QATemplateBundle"] = None,         # optional
) -> pd.DataFrame:
    model_cfg = model_cfg or ModelConfig()
    prompt_cfg = prompt_cfg or PromptConfig()
    rag_cfg = rag_cfg or RAGConfig()
    spec_templates = spec_templates or TemplateBundle()

    prompt_renderer = PromptRenderer(prompt_cfg.template_dir)
    agents = build_agents(
        model_cfg,
        prompt_renderer,
        spec_templates=spec_templates,
    )

    fileKVCache = FileKVCache(file_cache_path)
    returned_rows: List[Dict[str, Any]] = []

    for _, row in input_df.iterrows():
        def _call(**kwargs: Any) -> Dict[str, Any]:
            return build_prompt_call_pydantic_ai(
                agents,
                prompt_cfg,
                rag_cfg,
                prompt_renderer,
                templates=spec_templates,
                **kwargs,
            )

        # IMPORTANT: include templates in cache key to avoid stale reuse
        cache_key = "|".join(
            [
                row["filename"],
                model_cfg.model,
                spec_templates.spec_system,
                spec_templates.spec_user,
                input_mode,
            ]
        )
        if qa_templates is not None:
            cache_key += "|" + "|".join([qa_templates.qa_system, qa_templates.qa_user])

        result = fileKVCache.get_or_call(
            cache_key,
            _call,
            {
                "input_mode": input_mode,
                "filename": row["filename"],
                "lang": row["filetype"],
                "code": row["content"],
            },
        )

        returned_rows.append(
            {
                "filename": row["filename"],
                "filetype": row["filetype"],
                "set": row.get("set", ""),
                **result,
                "content": row["content"],
            }
        )

    return pd.DataFrame(returned_rows)
