from __future__ import annotations

import os
import re
import glob
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Tuple
import chromadb
from openai import OpenAI

# -----------------------------
# Chunking
# -----------------------------
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

# 
def chunk_markdown(md: str, source_path: str, max_chars: int = 1800, overlap: int = 200) -> List[Chunk]:
    """
    Simple chunker:
      - keeps headings (roughly) by splitting on markdown headings
      - then windows text into <= max_chars with overlap
    """
    md = strip_front_matter(md).replace("\r\n", "\n")
    # Split into sections by headings, but keep the heading in the section
    parts = re.split(r"(?m)^(#{1,6}\s+.*)$", md)
    sections: List[str] = []
    i = 0
    while i < len(parts):
        if parts[i].startswith("#"):
            heading = parts[i].strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            sections.append(f"{heading}\n{body}".strip())
            i += 2
        else:
            # pre-heading content
            if parts[i].strip():
                sections.append(parts[i].strip())
            i += 1

    chunks: List[Chunk] = []

    def make_id(text: str, extra: str) -> str:
        h = hashlib.sha256((extra + "\n" + text).encode("utf-8")).hexdigest()[:24]
        return h

    for sec_idx, sec in enumerate(sections):
        if not sec.strip():
            continue
        start = 0
        while start < len(sec):
            end = min(len(sec), start + max_chars)
            window = sec[start:end].strip()
            if window:
                cid = make_id(window, f"{source_path}:{sec_idx}:{start}")
                chunks.append(
                    Chunk(
                        id=cid,
                        text=window,
                        meta={
                            "source": source_path,
                            "section_index": sec_idx,
                            "start_char": start,
                        },
                    )
                )
            if end >= len(sec):
                break
            start = max(0, end - overlap)

    return chunks

# -----------------------------
# Embeddings
# -----------------------------
def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Batch embeddings. OpenAI embeddings endpoint supports array-of-strings input. :contentReference[oaicite:1]{index=1}
    """
    resp = client.embeddings.create(model=model, input=texts)
    # The API returns embeddings in the same order as inputs.
    return [item.embedding for item in resp.data]

# -----------------------------
# Index + Query with Chroma
# -----------------------------
def get_chroma_collection(persist_dir: str, name: str = "md_kb"):
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    return chroma_client.get_or_create_collection(name=name)

def index_markdown_folder(
    md_folder: str,
    persist_dir: str = "./chroma_db",
    collection_name: str = "md_kb",
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 64,
):
    """
    Reads markdown, chunks it, embeds each chunk, stores:
      - embeddings (vectors) for similarity search
      - documents (chunk text) so we can use them as context later
      - metadata (source info)
    """
    openai_client = OpenAI()
    col = get_chroma_collection(persist_dir, collection_name)

    docs = read_markdown_files(md_folder)
    all_chunks: List[Chunk] = []
    for path, text in docs:
        all_chunks.extend(chunk_markdown(text, source_path=path))

    # De-dup by chunk id (if rerun)
    # We'll check existing IDs in batches by trying "get"—simple approach: just upsert via add with unique IDs.
    # If you re-run often, consider storing an "indexed_at" and doing update/upsert.
    ids = [c.id for c in all_chunks]
    texts = [c.text for c in all_chunks]
    metas = [c.meta for c in all_chunks]

    # Embed + add in batches
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metas = metas[i:i + batch_size]

        vectors = embed_texts(openai_client, batch_texts, model=embedding_model)

        # Store embeddings AND the documents (chunk text)
        # Using embeddings you provide (rather than Chroma's embedding_function)
        col.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metas,
            embeddings=vectors,
        )

    print(f"Indexed {len(all_chunks)} chunks into Chroma collection '{collection_name}' at {persist_dir}")

def retrieve(
    query: str,
    persist_dir: str = "./chroma_db",
    collection_name: str = "md_kb",
    embedding_model: str = "text-embedding-3-small",
    k: int = 5,
):
    """
    Embeds the query, then does vector similarity search.
    Returns retrieved chunks (text) + metadata.
    """
    openai_client = OpenAI()
    col = get_chroma_collection(persist_dir, collection_name)

    qvec = embed_texts(openai_client, [query], model=embedding_model)[0]
    res = col.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances", "ids"],
    )

    hits = []
    for doc, meta, dist, _id in zip(res["documents"][0], res["metadatas"][0], res["distances"][0], res["ids"][0]):
        hits.append({"id": _id, "text": doc, "meta": meta, "distance": dist})
    return hits

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

# -----------------------------
# Example "use chunks as context"
# -----------------------------
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
    # index_markdown_folder("./my_markdown_kb")

    # 2) Query + retrieve top-k chunks
    q = "How do I define a role in a CPSA protocol?"
    hits = retrieve(q, k=5)

    print("\nTop hits:")
    for h in hits:
        print(f"- {h['distance']:.4f} :: {h['meta']['source']} (chunk_id={h['id']})")

    # 3) Use retrieved chunks as context
    # ans = answer_with_context(q, hits)
    # print("\nAnswer:\n", ans)
