# cluster.py (notebook-friendly)
# pip install pandas numpy
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from collections import defaultdict, deque

# -------- defaults (can be overridden via run_clustering args) --------
DEFAULT_JACCARD_KEEP_THRESHOLD = 0.70
DEFAULT_OUT_DIR = "minhash_outputs"

# --- helper: build clusters (connected components) via BFS ---
def build_clusters(df_pairs: pd.DataFrame, threshold: float) -> List[List[str]]:
    """
    Build connected components using edges with jaccard >= threshold.
    Returns a list of components; each component is a list of filenames.
    """
    adj = defaultdict(set)
    for _, row in df_pairs.iterrows():
        if row["jaccard"] >= threshold:
            a, b = row["a"], row["b"]
            adj[a].add(b)
            adj[b].add(a)

    nodes = set(df_pairs["a"]).union(set(df_pairs["b"]))
    return list(_connected_components(adj, nodes))

def _connected_components(adj: Dict[str, set], nodes: Iterable[str]):
    seen = set()
    for n in nodes:
        if n in seen:
            continue
        comp = []
        q = deque([n])
        seen.add(n)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        yield comp

# --- quality scoring helpers (unchanged, just wrapped) ---
_junk_path_re = re.compile(
    r"(?:^|/)(?:test|tests|example|examples|tmp|backup|backups|copy|copies|vendor|third_party|old)(?:/|$)",
    re.I,
)

def hex_num_ratio_from_tokens(num_tokens, text):
    if not text:
        return 0.0
    long_hex = len(re.findall(r"0x[0-9A-Fa-f]{4,}", text))
    long_nums = len(re.findall(r"\b\d{6,}\b", text))
    denom = max(1, int(num_tokens) if pd.notna(num_tokens) else 1)
    return min(1.0, (long_hex + long_nums) / denom)

def non_ascii_ratio(text):
    if not text:
        return 0.0
    non = sum(1 for ch in text if ord(ch) > 127)
    return non / len(text) if text else 0.0

def sweet_spot_score(n, low=50, high=5000):
    try:
        n = float(n)
    except Exception:
        return 0.0
    if n <= 0:
        return 0.0
    if n < low:
        return (n / low) * 0.5
    if n > high:
        return max(0.0, 1.0 - (np.log1p(n - high) / np.log1p(high)))
    return 1.0

def file_quality_score(row: pd.Series, raw_text: Optional[str] = None,
                       parse_ok: Optional[bool] = None, proof_ok: Optional[bool] = None) -> float:
    score = 0.0

    if parse_ok is True:
        score += 3.0
    if proof_ok is True:
        score += 3.0

    def getf(key, default=0.0):
        try:
            return float(row[key])
        except Exception:
            return float(default)

    size_tok = sweet_spot_score(getf("num_tokens"))
    size_shg = sweet_spot_score(getf("num_shingles"))
    score += 3.0 * size_tok
    score += 2.0 * size_shg

    try:
        path_val = row["filename"]
    except Exception:
        path_val = row.name

    if isinstance(path_val, str):
        path_norm = path_val.replace("\\", "/")
        if _junk_path_re.search(path_norm):
            score -= 2.0

    if raw_text is not None:
        score -= 2.0 * hex_num_ratio_from_tokens(getf("num_tokens"), raw_text)
        score -= 1.0 * non_ascii_ratio(raw_text)

    return float(score)

# -------- main entry (call this from your notebook) --------
def run_clustering(
    df_files: Optional[pd.DataFrame] = None,
    df_pairs: Optional[pd.DataFrame] = None,
    *,
    files_csv: str = os.path.join(DEFAULT_OUT_DIR, "minhash_files.csv"),
    pairs_csv: str = os.path.join(DEFAULT_OUT_DIR, "minhash_pairs.csv"),
    jaccard_keep_threshold: float = DEFAULT_JACCARD_KEEP_THRESHOLD,
    out_dir: str = DEFAULT_OUT_DIR,
    content_lookup: Optional[Dict[str, str]] = None,  # optional raw text for penalties
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cluster near-duplicate files using Jaccard pairs, select a representative per cluster,
    and (optionally) write dedup CSVs.

    Parameters
    ----------
    df_files : DataFrame with at least ["filename","num_tokens","num_shingles"]
               If None, loads from files_csv.
    df_pairs : DataFrame with at least ["a","b","jaccard"]
               If None, loads from pairs_csv.
    files_csv, pairs_csv : Paths to load if df_* not supplied.
    jaccard_keep_threshold : Edge threshold to connect files in a cluster.
    out_dir : Where to save outputs if save_outputs=True.
    content_lookup : Optional dict {filename -> raw_text} used in scoring.
    save_outputs : If True, writes dedup_keep.csv, dedup_drop.csv, dedup_clusters.csv
                   and train_files_after_dedup.csv to out_dir.

    Returns
    -------
    (df_keep, df_drop, df_clusters)
    """
    os.makedirs(out_dir, exist_ok=True)

    if df_files is None:
        df_files = pd.read_csv(files_csv)
    if df_pairs is None:
        df_pairs = pd.read_csv(pairs_csv)

    # Ensure every file appears at least as a singleton cluster
    all_paths = set(df_files["filename"])
    pair_nodes = set(df_pairs["a"]).union(set(df_pairs["b"]))
    clusters = build_clusters(df_pairs, jaccard_keep_threshold)

    singleton_paths = sorted(all_paths - pair_nodes)
    for p in singleton_paths:
        clusters.append([p])

    if content_lookup is None:
        content_lookup = {}

    file_index = df_files.set_index("filename")
    keep_rows, drop_rows, cluster_rows = [], [], []

    for cid, comp in enumerate(clusters, start=1):
        scored = []
        for p in comp:
            if p not in file_index.index:
                continue
            row = file_index.loc[p]
            raw = content_lookup.get(p)  # None if unknown
            s = file_quality_score(row, raw_text=raw)
            scored.append((p, s, row.get("num_shingles", 0), row.get("num_tokens", 0), len(p)))

        if not scored:
            continue

        #scored.sort(key=lambda t: (-t[1], -t[2], -t[3], t[4], t[0]))
        # Keep the longest always (by tokens), regardless of quality score
        scored.sort(key=lambda t: (-t[3], -t[2], -t[1], t[4], t[0]))
        #                     ^num_tokens  ^num_shingles ^score

        keep = scored[0][0]
        drops = [p for p, *_ in scored[1:]]

        keep_rows.append({"cluster_id": cid, "filename": keep, "score": float(scored[0][1])})
        for p, s, ns, nt, _ in scored[1:]:
            drop_rows.append({"cluster_id": cid, "filename": p, "score": float(s)})

        cluster_rows.append({
            "cluster_id": cid,
            "size": len(comp),
            "kept": keep,
            "dropped_count": len(drops),
        })

    df_keep  = pd.DataFrame(keep_rows).sort_values(["cluster_id", "filename"]).reset_index(drop=True)
    df_drop  = pd.DataFrame(drop_rows).sort_values(["cluster_id", "filename"]).reset_index(drop=True)
    df_clust = pd.DataFrame(cluster_rows).sort_values("cluster_id").reset_index(drop=True)

    print(f"[info] clusters formed   : {len(clusters)}")
    print(f"[info] kept files        : {len(df_keep)}")
    print(f"[info] dropped files     : {len(df_drop)}")

    if save_outputs:
        df_keep.to_csv(os.path.join(out_dir, "dedup_keep.csv"), index=False)
        df_drop.to_csv(os.path.join(out_dir, "dedup_drop.csv"), index=False)
        df_clust.to_csv(os.path.join(out_dir, "dedup_clusters.csv"), index=False)

        keep_set = set(df_keep["filename"])
        df_train_files = df_files[df_files["filename"].isin(keep_set)].copy()
        df_train_files.to_csv(os.path.join(out_dir, "train_files_after_dedup.csv"), index=False)
        print(f"[info] wrote keep/drop/cluster CSVs to {out_dir}/")

    return df_keep, df_drop, df_clust

# ----- optional CLI for script use (safe in notebook; only runs if __main__) -----
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Cluster similar files from MinHash pairs and select representatives.")
    p.add_argument("--files-csv", default=os.path.join(DEFAULT_OUT_DIR, "minhash_files.csv"))
    p.add_argument("--pairs-csv", default=os.path.join(DEFAULT_OUT_DIR, "minhash_pairs.csv"))
    p.add_argument("--threshold", type=float, default=DEFAULT_JACCARD_KEEP_THRESHOLD)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    run_clustering(
        df_files=None,
        df_pairs=None,
        files_csv=args.files_csv,
        pairs_csv=args.pairs_csv,
        jaccard_keep_threshold=args.threshold,
        out_dir=args.out_dir,
        save_outputs=not args.no_save,
    )
