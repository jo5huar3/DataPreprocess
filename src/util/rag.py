from __future__ import annotations
import os
import re
import glob
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Tuple
import chromadb
from openai import OpenAI

# Chunk(id, text, meta)
@dataclass
class Chunk:
    id: str
    text: str
    meta: Dict[str, Any]

# Recursively read markdown files from a folder. Return list of (path, text).
def read_markdown_files(folder: str) -> List[Tuple[str, str]]:
    """Return list of (path, text)."""
    paths = sorted(glob.glob(os.path.join(folder, "**/*.md"), recursive=True))
    docs: List[Tuple[str, str]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            docs.append((p, f.read()))
    return docs

# Remove YAML front matter from markdown. Example YAML: --- ... ---
def strip_front_matter(md: str) -> str:
    # Remove YAML front matter if present
    if md.startswith("---"):
        m = re.match(r"^---\s*\n.*?\n---\s*\n", md, flags=re.DOTALL)
        if m:
            return md[m.end():]
    return md

def chunk_markdown(
    md: str,
    source_path: str,
    max_chars: int = 1800,
    overlap: int = 200,
    min_chars: int = 600,  # threshold for "too small, try to merge forward"
) -> List[Chunk]:
    """
    Greedy heading packer:
      1) Split markdown into heading sections (heading + following body).
      2) Pack sections linearly into chunks up to max_chars (greedy).
      3) Enforce min_chars by preferring to merge small chunks forward when possible.
      4) If a single section > max_chars, split it internally using sliding windows,
         but ALWAYS keep the heading at the top of each window.
    """
    md = strip_front_matter(md).replace("\r\n", "\n")

    # Split into sections by headings, keeping headings
    parts = re.split(r"(?m)^(#{1,6}\s+.*)$", md)
    sections: List[str] = []
    i = 0
    while i < len(parts):
        if parts[i].startswith("#"):
            heading = parts[i].strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            sec = f"{heading}\n{body}".strip()
            if sec:
                sections.append(sec)
            i += 2
        else:
            # pre-heading content
            if parts[i].strip():
                sections.append(parts[i].strip())
            i += 1

    def make_id(text: str, extra: str) -> str:
        return hashlib.sha256((extra + "\n" + text).encode("utf-8")).hexdigest()[:24]

    def split_oversized_section(sec: str, sec_idx: int) -> List[Chunk]:
        """
        If a single heading section is too large, window it, but keep the heading line.
        """
        lines = sec.splitlines()
        heading_line = lines[0].strip() if lines and lines[0].lstrip().startswith("#") else ""
        rest = "\n".join(lines[1:]).strip() if heading_line else sec

        prefix = (heading_line + "\n").strip() if heading_line else ""
        prefix_len = len(prefix) + (1 if prefix else 0)

        payload = rest
        chunks_local: List[Chunk] = []
        start = 0
        # available space for payload after prefix
        payload_max = max(1, max_chars - prefix_len)

        while start < len(payload):
            end = min(len(payload), start + payload_max)
            window_payload = payload[start:end].strip()
            if window_payload:
                text = (prefix + "\n" + window_payload).strip() if prefix else window_payload
                cid = make_id(text, f"{source_path}:{sec_idx}:{start}")
                chunks_local.append(
                    Chunk(
                        id=cid,
                        text=text,
                        meta={
                            "source": source_path,
                            "section_start": sec_idx,
                            "section_end": sec_idx,
                            "start_char": start,
                        },
                    )
                )
            if end >= len(payload):
                break
            start = max(0, end - overlap)

        return chunks_local

    # Greedy packing of sections into chunks
    chunks: List[Chunk] = []
    cur_parts: List[str] = []
    cur_start_sec: int = 0

    def flush(cur_end_sec: int):
        nonlocal cur_parts, cur_start_sec, chunks
        if not cur_parts:
            return
        text = "\n\n".join(cur_parts).strip()
        if not text:
            cur_parts = []
            return
        cid = make_id(text, f"{source_path}:{cur_start_sec}:{cur_end_sec}")
        chunks.append(
            Chunk(
                id=cid,
                text=text,
                meta={
                    "source": source_path,
                    "section_start": cur_start_sec,
                    "section_end": cur_end_sec,
                },
            )
        )
        cur_parts = []

    for sec_idx, sec in enumerate(sections):
        if not sec.strip():
            continue

        # If section itself is huge, flush current and split this one
        if len(sec) > max_chars:
            # flush what we have first
            if cur_parts:
                flush(sec_idx - 1)

            # split oversized section into windows (heading preserved)
            chunks.extend(split_oversized_section(sec, sec_idx))
            continue

        if not cur_parts:
            cur_parts = [sec]
            cur_start_sec = sec_idx
            continue

        candidate = "\n\n".join(cur_parts + [sec]).strip()

        # Greedy rule: keep adding as long as it fits max_chars
        if len(candidate) <= max_chars:
            cur_parts.append(sec)
            continue

        # Would exceed max_chars: flush current
        # BUT if current chunk is tiny (< min_chars), try to avoid tiny chunks by
        # flushing anyway (we have no room), and start new with sec.
        flush(sec_idx - 1)
        cur_parts = [sec]
        cur_start_sec = sec_idx

    # flush leftover
    if cur_parts:
        flush(len(sections) - 1)

    # Optional: drop chunks that are basically only headings (very low signal)
    cleaned: List[Chunk] = []
    for c in chunks:
        lines = [ln for ln in c.text.splitlines() if ln.strip()]
        non_heading = [ln for ln in lines if not re.match(r"^#{1,6}\s+", ln.strip())]
        if len(" ".join(non_heading).strip()) < 80:
            # skip near-empty content (mostly headings)
            continue
        cleaned.append(c)

    # Second pass: merge tiny chunks forward if they ended up < min_chars and merging fits.
    # (Linear, greedy; preserves order.)
    merged: List[Chunk] = []
    i = 0
    while i < len(cleaned):
        cur = cleaned[i]
        if len(cur.text) >= min_chars or i == len(cleaned) - 1:
            merged.append(cur)
            i += 1
            continue

        nxt = cleaned[i + 1]
        merged_text = (cur.text + "\n\n" + nxt.text).strip()
        if len(merged_text) <= max_chars:
            cid = make_id(merged_text, f"{source_path}:{cur.meta['section_start']}:{nxt.meta['section_end']}")
            merged.append(
                Chunk(
                    id=cid,
                    text=merged_text,
                    meta={
                        "source": source_path,
                        "section_start": cur.meta["section_start"],
                        "section_end": nxt.meta["section_end"],
                    },
                )
            )
            i += 2
        else:
            merged.append(cur)
            i += 1

    return merged

# Create embeddings for a list of texts.
def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Batch embeddings. OpenAI embeddings endpoint supports array-of-strings input. :contentReference[oaicite:1]{index=1}
    """
    resp = client.embeddings.create(model=model, input=texts)
    # The API returns embeddings in the same order as inputs.
    return [item.embedding for item in resp.data]

# Vector store setup. Create persistant vector store or return existing one.
def get_chroma_collection(persist_dir: str, name: str = "md_kb"):
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    return chroma_client.get_or_create_collection(name=name)

# Index a single markdown text document. 
def index_markdown_text(
    *,
    md_text: str,
    source_path: str,
    collection,
    openai_client: OpenAI,
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 64,
    chunk_max_chars: int = 1800,
    chunk_overlap: int = 200,
    # If you want to avoid "ID already exists" errors when re-indexing the same file:
    skip_existing: bool = True,
) -> int:
    """
    Index a SINGLE markdown document provided as a string.

    Inputs:
      - md_text: the markdown content
      - source_path: identifier for metadata + stable chunk IDs (can be a real path or logical name)
      - collection: a Chroma collection
      - openai_client: initialized OpenAI client
      - embedding_model: OpenAI embedding model
      - batch_size: embedding/insert batch size
      - chunk_max_chars / chunk_overlap: chunking params
      - skip_existing: if True, checks IDs before adding to avoid duplicates

    Output:
      - int: number of chunks added (not total chunks produced)
    """
    # 1) chunk the markdown string
    chunks = chunk_markdown(
        md_text,
        source_path=source_path,
        max_chars=chunk_max_chars,
        overlap=chunk_overlap,
    )

    if not chunks:
        return 0

    # 2) prepare lists
    ids = [c.id for c in chunks]
    texts = [c.text for c in chunks]
    metas = [c.meta for c in chunks]

    added = 0

    # 3) embed + store in batches
    for i in range(0, len(chunks), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        batch_metas = metas[i : i + batch_size]

        # Optionally filter out IDs that already exist in the DB
        if skip_existing:
            existing = collection.get(ids=batch_ids, include=[])
            existing_ids = set(existing.get("ids", []))
            if existing_ids:
                keep = [
                    j for j, _id in enumerate(batch_ids)
                    if _id not in existing_ids
                ]
                if not keep:
                    continue
                batch_ids = [batch_ids[j] for j in keep]
                batch_texts = [batch_texts[j] for j in keep]
                batch_metas = [batch_metas[j] for j in keep]

        # If nothing left after filtering, skip
        if not batch_ids:
            continue

        vectors = embed_texts(openai_client, batch_texts, model=embedding_model)

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metas,
            embeddings=vectors,
        )
        added += len(batch_ids)

    return added

# Index all markdown files under a directory.
def index_markdown_folder(
    md_folder: str,
    persist_dir: str = "./chroma_db",
    collection_name: str = "md_kb",
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 64,
    chunk_max_chars: int = 1800,
    chunk_overlap: int = 200,
    skip_existing: bool = True,
) -> None:
    """
    Reads markdown files in a folder and indexes each file by calling index_markdown_text().
    """
    openai_client = OpenAI()
    col = get_chroma_collection(persist_dir, collection_name)

    docs = read_markdown_files(md_folder)

    total_added = 0
    for path, text in docs:
        added = index_markdown_text(
            md_text=text,
            source_path=path,
            collection=col,
            openai_client=openai_client,
            embedding_model=embedding_model,
            batch_size=batch_size,
            chunk_max_chars=chunk_max_chars,
            chunk_overlap=chunk_overlap,
            skip_existing=skip_existing,
        )
        total_added += added

    print(
        f"Indexed {total_added} new chunks into Chroma collection '{collection_name}' at {persist_dir}"
    )

# Create embedding for the query and return k nearest neighbors from vector the database.
def retrieve(
    query: str,
    persist_dir: str = "./chroma_db",
    collection_name: str = "md_kb",
    embedding_model: str = "text-embedding-3-small",
    k: int = 5,
):
    openai_client = OpenAI()
    col = get_chroma_collection(persist_dir, collection_name)

    qvec = embed_texts(openai_client, [query], model=embedding_model)[0]
    res = col.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],  # ✅ remove "ids"
    )

    hits = []
    for doc, meta, dist, _id in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
        res["ids"][0],  # ✅ ids still returned automatically
    ):
        hits.append({"id": _id, "text": doc, "meta": meta, "distance": dist})
    return hits

# Convert list of chunks into a single context string.
def build_context(hits: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """
    Concatenate retrieved chunk texts into one context block.
    """
    out = []
    used = 0
    for h in hits:
        snippet = f"SOURCE: {h['meta']['source']}\n{h['text']}".strip()
        if used + len(snippet) > max_chars:
            break
        out.append(snippet)
        used += len(snippet) + 2
    return "\n\n---\n\n".join(out)

# Answer a query using the retrieved context.
def answer_with_context(query: str, hits: List[Dict[str, Any]]):
    """
    This shows the LAST PART you were confused about:
    - embeddings are used only to retrieve
    - the *retrieved chunk text* is what you pass to the LLM as context
    """
    client = OpenAI()
    context = build_context(hits)

    # You can use Chat Completions or Responses API; this is just a minimal example.
    # (Model name here is an example; pick what you use in your project.)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": "Answer ONLY using the provided context. If missing, say you don't know.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}",
            },
        ],
    )
    return resp.output_text

if __name__ == "__main__":
    # 1) Index markdown folder
    index_markdown_folder("/Users/josh/Automated_Reasoning_for_Cryptography/DataPreprocess/text/Cryptol-Reference-Manual-MD")

    # 2) Query + retrieve top-k chunks
    q = "How do I define a polymorphic type?"
    hits = retrieve(q, k=5)

    print(q)
    print("\nTop hits:")
    for h in hits:
        print(f"- {h['distance']:.4f} :: {h['meta']['source']} (chunk_id={h['id']})")
        print(h['text'], "\n" + '-' * 80)

    # 3) Use retrieved chunks as context
    # ans = answer_with_context(q, hits)
    # print("\nAnswer:\n", ans)
