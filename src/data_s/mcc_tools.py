from __future__ import annotations
import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import pandas as pd

# -----------------------------
# Core MCC computation utilities
# -----------------------------
CONTROL_EDGE_KINDS = {"control", "branch", "back"}

def _reachable_from_entry(mcc_block: Dict[str, Any]) -> Set[int]:
    """
    Robustly restrict the MCC graph to real control-flow:
      - only edges of kind in CONTROL_EDGE_KINDS
      - only nodes reachable from entry via those edges

    This prevents meta edges like "declares" from polluting node/edge counts.
    """
    nodes = mcc_block.get("nodes", []) or []
    edges = mcc_block.get("edges", []) or []
    node_ids = {n["id"] for n in nodes if "id" in n}

    entry = mcc_block.get("entry")
    if entry is None or entry not in node_ids:
        return set()

    adj = defaultdict(list)
    for e in edges:
        if e.get("kind") in CONTROL_EDGE_KINDS:
            u, v = e.get("from"), e.get("to")
            if u in node_ids and v in node_ids:
                adj[u].append(v)

    seen = {entry}
    q = deque([entry])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen

def compute_mcc_robust(mcc_block: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Same MCC formula, but computed on the reachable control-flow subgraph only.
    Returns (M, E, N, P).
    """
    if not mcc_block:
        return None, None, None, None

    keep = _reachable_from_entry(mcc_block)
    if not keep:
        return None, None, None, None

    nodes_all = mcc_block.get("nodes", []) or []
    edges_all = mcc_block.get("edges", []) or []

    nodes = [n for n in nodes_all if n.get("id") in keep]
    edges = [
        e for e in edges_all
        if e.get("kind") in CONTROL_EDGE_KINDS and e.get("from") in keep and e.get("to") in keep
    ]

    E = len(edges)
    N = len(nodes)
    P = connected_components_count(nodes, edges) if nodes else 0
    M = E - N + 2 * P if P else None
    return M, E, N, P

def iter_definitions_with_prefix(json_obj: Dict[str, Any], base_name: str) -> Iterable[Dict[str, Any]]:
    """
    Yield the definition whose name == base_name AND every definition that starts with f"{base_name}::".
    Uses "::" boundary so 'pmult2' won't match 'pmult'.
    """
    prefix = base_name + "::"
    for d in (json_obj.get("definitions", []) or []):
        name = d.get("name") or ""
        if name == base_name or name.startswith(prefix):
            yield d

def definition_stats_row(d: Dict[str, Any], json_path: Optional[Union[str, Path]] = None, base: Optional[str] = None) -> Dict[str, Any]:
    name = d.get("name")
    M, E, N, P = compute_mcc_robust(d.get("mcc"))  # <-- robust MCC

    return {
        "json_path": str(json_path) if json_path is not None else None,
        "base": base,
        "definition_name": name,
        "depth": (name.count("::") if isinstance(name, str) else None),
        "kind": d.get("kind"),
        "signature": d.get("signature"),
        "mcc": M,
        "edges_E": E,
        "nodes_N": N,
        "components_P": P,
        "num_params": len(d.get("params") or []),
        "num_locals": len(d.get("locals") or []),
        "num_references": len(d.get("references") or []),
    }

def definition_group_stats_df(
    json_obj: Dict[str, Any],
    base_name: str,
    json_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    One row per definition in the prefix group.
    """
    rows = [definition_stats_row(d, json_path=json_path, base=base_name) for d in iter_definitions_with_prefix(json_obj, base_name)]
    df = pd.DataFrame(rows)

    if not df.empty:
        df["mcc"] = pd.to_numeric(df["mcc"], errors="coerce")
        df = df.sort_values(["depth", "definition_name"], ascending=[True, True]).reset_index(drop=True)

    return df

def get_hardest_definition(mcc_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get the hardest definition from the DataFrame.
    """
    rows = []
    for d in mcc_json.get("definitions", []):
        if d.get("kind") != "declaration" or not d.get("name"):
            continue
        base = d["name"].split("::", 1)[0]
        M, *_ = compute_mcc_robust(d.get("mcc"))
        rows.append({"base": base, "mcc": int(M or 0)})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    hardest = df.groupby("base")["mcc"].sum().sort_values(ascending=False)
    hardest_name = hardest.index[0] if not hardest.empty else None
    return hardest_name if not hardest_name else find_definition(mcc_json, hardest_name)

def summarize_definition_group(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summary for a prefix group.
    Includes:
      - sum_mcc: sum of MCC scores of all definitions in the group
      - inline_mcc: 1 + sum(M_i - 1)  (good for "top-level + where locals")
    """
    if df.empty:
        return {"num_defs": 0, "sum_mcc": None, "inline_mcc": None, "max_mcc": None, "avg_mcc": None}

    m = df["mcc"].dropna()
    if m.empty:
        return {"num_defs": int(len(df)), "sum_mcc": None, "inline_mcc": None, "max_mcc": None, "avg_mcc": None}

    m = m.astype(int)
    inline = int(1 + (m - 1).sum())
    return {
        "num_defs": int(len(df)),
        "sum_mcc": int(m.sum()),
        "inline_mcc": inline,
        "max_mcc": int(m.max()),
        "avg_mcc": float(m.mean()),
    }

def connected_components_count(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> int:
    """
    Count connected components P in the CFG using an undirected view.
    """
    node_ids: Set[int] = {n["id"] for n in nodes if "id" in n}

    adj: Dict[int, Set[int]] = defaultdict(set)
    for e in edges:
        u, v = e.get("from"), e.get("to")
        if u in node_ids and v in node_ids:
            adj[u].add(v)
            adj[v].add(u)

    seen: Set[int] = set()
    comps = 0
    for nid in node_ids:
        if nid in seen:
            continue
        comps += 1
        q = deque([nid])
        seen.add(nid)
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)

    # If there are nodes but no edges, P == number of isolated nodes (per the formula).
    return comps

def compute_mcc(mcc_block: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    McCabe Cyclomatic Complexity:
      M = E - N + 2P

    Returns (M, E, N, P). If mcc_block is missing/None or has no nodes, returns (None, None, None, None).
    """
    if not mcc_block:
        return None, None, None, None

    edges = mcc_block.get("edges", []) or []
    nodes = mcc_block.get("nodes", []) or []
    if not nodes:
        return None, len(edges), 0, None

    E = len(edges)
    N = len(nodes)
    P = connected_components_count(nodes, edges)
    M = E - N + 2 * P
    return M, E, N, P

def find_definition(json_obj: Dict[str, Any], definition_name: str) -> Dict[str, Any]:
    """
    Find a definition by exact name in json_obj["definitions"].
    Raises KeyError if not found.
    """
    for d in json_obj.get("definitions", []) or []:
        if d.get("name") == definition_name:
            return d
    raise KeyError(f"Definition '{definition_name}' not found.")

def definition_mcc_from_obj(json_obj: Dict[str, Any], definition_name: str) -> Optional[int]:
    """
    Return MCC score for one definition name from a parsed json object.
    """
    d = find_definition(json_obj, definition_name)
    M, _, _, _ = compute_mcc(d.get("mcc"))
    return M

def definition_mcc_from_json_str(json_str: str, definition_name: str) -> Optional[int]:
    """
    Return MCC score for one definition name from a JSON string.
    """
    obj = json.loads(json_str)
    return definition_mcc_from_obj(obj, definition_name)

# -----------------------------
# CFG ("MCC graph") extraction
# -----------------------------
def _nodes_by_id(mcc_block: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    nodes = mcc_block.get("nodes", []) or []
    return {n["id"]: n for n in nodes if "id" in n}

def cfg_subgraph(
    mcc_block: Dict[str, Any],
    keep_node_ids: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """
    Return a pruned CFG (nodes/edges/entry/exit) containing only keep_node_ids (if provided).
    Keeps only edges where both endpoints are in keep_node_ids.
    """
    nodes = mcc_block.get("nodes", []) or []
    edges = mcc_block.get("edges", []) or []
    entry = mcc_block.get("entry")
    exit_ = mcc_block.get("exit")

    if keep_node_ids is None:
        return {"entry": entry, "exit": exit_, "nodes": nodes, "edges": edges}

    kept_nodes = [n for n in nodes if n.get("id") in keep_node_ids]
    kept_edges = [e for e in edges if e.get("from") in keep_node_ids and e.get("to") in keep_node_ids]

    # If entry/exit were pruned out, preserve them as None to make it explicit.
    if entry not in keep_node_ids:
        entry = None
    if exit_ not in keep_node_ids:
        exit_ = None

    return {"entry": entry, "exit": exit_, "nodes": kept_nodes, "edges": kept_edges}

def cfg_spanning_tree(
    mcc_block: Dict[str, Any],
    start: Union[int, str, None] = "entry",
    max_depth: Optional[int] = None,
    include_node_info: bool = True,
) -> Dict[str, Any]:
    """
    Build a directed spanning tree (BFS) over the CFG starting from:
      - "entry" (default)
      - "exit"
      - a specific node id (int)
      - None -> uses entry if available else smallest node id

    Returns a nested structure:
      {"root": <node_id>, "tree": {node_id: [child_id, ...]}, "nodes": {id: node_info} (optional)}
    """
    edges = mcc_block.get("edges", []) or []
    nodes = mcc_block.get("nodes", []) or []
    id_to_node = _nodes_by_id(mcc_block)

    if not nodes:
        return {"root": None, "tree": {}, "nodes": {} if include_node_info else None}

    entry = mcc_block.get("entry")
    exit_ = mcc_block.get("exit")

    if start == "entry":
        root = entry
    elif start == "exit":
        root = exit_
    elif isinstance(start, int):
        root = start
    else:
        # fallback
        root = entry if entry is not None else min(id_to_node.keys())

    if root is None or root not in id_to_node:
        root = min(id_to_node.keys())

    adj: Dict[int, List[int]] = defaultdict(list)
    for e in edges:
        u, v = e.get("from"), e.get("to")
        if u in id_to_node and v in id_to_node:
            adj[u].append(v)

    visited: Set[int] = set([root])
    tree: Dict[int, List[int]] = defaultdict(list)

    q = deque([(root, 0)])
    while q:
        cur, depth = q.popleft()
        if max_depth is not None and depth >= max_depth:
            continue

        for nxt in adj.get(cur, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            tree[cur].append(nxt)
            q.append((nxt, depth + 1))

    out = {"root": root, "tree": dict(tree)}
    if include_node_info:
        out["nodes"] = {nid: id_to_node[nid] for nid in visited}
    return out

# -----------------------------
# Definition dependency ("MCC tree") extraction
# -----------------------------
def _definition_dependency_map(json_obj: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Builds a map:
        def_name -> set(of referenced def names)
    using items in `references` with {"def": "..."}.

    This is NOT the CFG; it is the inter-definition dependency graph.
    """
    dep: Dict[str, Set[str]] = defaultdict(set)
    for d in json_obj.get("definitions", []) or []:
        name = d.get("name")
        if not name:
            continue

        for ref in d.get("references", []) or []:
            callee = ref.get("def")
            if callee:
                dep[name].add(callee)

    return dep

def definition_mcc_tree(
    json_obj: Dict[str, Any],
    roots: List[str],
    max_depth: Optional[int] = None,
    only_return: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Builds a dependency tree between *definitions* starting at each root, annotated with MCC.

    - roots: list of definition names to build trees for
    - max_depth: limit depth (None = unlimited)
    - only_return: if provided, restricts the returned nodes to this set (roots are always included)

    Returns:
      {
        "<root>": {"mcc": <int|None>, "children": { ... }},
        ...
      }

    Cycles are cut off using a visited set per root.
    """
    dep = _definition_dependency_map(json_obj)

    # Precompute MCC per definition for quick annotation.
    mcc_map: Dict[str, Optional[int]] = {}
    for d in json_obj.get("definitions", []) or []:
        if d.get("kind") != "declaration":
            continue
        name = d.get("name")
        if not name:
            continue
        M, _, _, _ = compute_mcc(d.get("mcc"))
        mcc_map[name] = M

    def build(node: str, depth: int, seen: Set[str]) -> Dict[str, Any]:
        out = {"mcc": mcc_map.get(node), "children": {}}
        if max_depth is not None and depth >= max_depth:
            return out

        for child in sorted(dep.get(node, [])):
            if child in seen:
                # cycle cutoff
                continue
            seen.add(child)
            out["children"][child] = build(child, depth + 1, seen)
        return out

    result: Dict[str, Any] = {}
    for r in roots:
        if r not in dep and r not in mcc_map:
            # still return it, but likely not present
            result[r] = {"mcc": None, "children": {}}
            continue
        result[r] = build(r, 0, seen=set([r]))

    if only_return is not None:
        only_return = set(only_return) | set(roots)

        def prune(tree_node: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
            kept_children = {}
            for child_name, child_tree in tree_node.get("children", {}).items():
                pruned = prune(child_tree, child_name)
                if pruned is not None:
                    kept_children[child_name] = pruned

            if name in only_return or kept_children:
                return {"mcc": tree_node.get("mcc"), "children": kept_children}
            return None

        pruned_result = {}
        for r, t in result.items():
            pruned = prune(t, r)
            pruned_result[r] = pruned if pruned is not None else {"mcc": None, "children": {}}
        result = pruned_result

    return result

# -----------------------------
# Summary indexing (1 row per JSON file)
# -----------------------------
def summarize_file_obj(json_obj: Dict[str, Any]) -> Dict[str, Any]:
    defs = json_obj.get("definitions", []) or []

    decl_scores: List[int] = []
    decl_with_mcc = 0
    for d in defs:
        if d.get("kind") != "declaration":
            continue
        M, _, _, _ = compute_mcc(d.get("mcc"))
        if M is not None:
            decl_scores.append(M)
            decl_with_mcc += 1

    num_decls = sum(1 for d in defs if d.get("kind") == "declaration")
    num_types = sum(1 for d in defs if d.get("kind") == "type")

    return {
        "imports": json_obj.get("imports", []) or [],
        "imports_count": len(json_obj.get("imports", []) or []),
        "num_definitions": len(defs),
        "num_declarations": num_decls,
        "num_types": num_types,
        "num_declarations_with_mcc": decl_with_mcc,
        "total_mcc": int(sum(decl_scores)) if decl_scores else 0,
        "max_mcc": int(max(decl_scores)) if decl_scores else None,
        "avg_mcc": (float(sum(decl_scores)) / float(len(decl_scores))) if decl_scores else None,
    }

@lru_cache(maxsize=64)
def _load_json_cached(path_str: str, mtime_ns: int) -> Dict[str, Any]:
    """
    Cached loader: cache key includes file mtime_ns so edits invalidate cache automatically.
    """
    p = Path(path_str)
    return json.loads(p.read_text(encoding="utf-8"))

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    st = p.stat()
    return _load_json_cached(str(p), st.st_mtime_ns)

def build_summary_df(
    json_paths: Iterable[Union[str, Path]],
    include_imports_as_json_str: bool = True,
) -> pd.DataFrame:
    """
    Build a 1-row-per-JSON summary dataframe.
    """
    rows = []
    for p in map(Path, json_paths):
        obj = load_json(p)
        s = summarize_file_obj(obj)

        row = {
            "json_path": str(p),
            "file_size": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
            **s,
        }
        if include_imports_as_json_str:
            row["imports"] = json.dumps(row["imports"], ensure_ascii=False)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Nice default sort: most complex files first
    if not df.empty and "total_mcc" in df.columns:
        df = df.sort_values("total_mcc", ascending=False)

    return df

def build_definition_index_df(json_paths: Iterable[Union[str, Path]]) -> pd.DataFrame:
    """
    OPTIONAL: Build a per-definition index for fast querying across many files.

    Rows: (json_path, definition_name, kind, mcc, E, N, P, signature)
    """
    rows = []
    for p in map(Path, json_paths):
        obj = load_json(p)
        for d in obj.get("definitions", []) or []:
            name = d.get("name")
            kind = d.get("kind")
            if not name:
                continue

            M, E, N, P = compute_mcc(d.get("mcc"))
            rows.append({
                "json_path": str(p),
                "definition_name": name,
                "kind": kind,
                "mcc": M,
                "edges_E": E,
                "nodes_N": N,
                "components_P": P,
                "signature": d.get("signature"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["mcc"] = pd.to_numeric(df["mcc"], errors="coerce")
        df = df.sort_values(["mcc", "json_path", "definition_name"], ascending=[False, True, True])
    return df

# -----------------------------
# High-level convenience class
# -----------------------------
@dataclass
class MCCCorpus:
    """
    Manage a collection of MCC JSON files:
      - build summary DF (1 row per file)
      - compute MCC of a definition by name
      - return CFG graphs/trees for a definition
      - build definition dependency trees annotated with MCC
    """
    json_paths: List[Path]

    @classmethod
    def from_dir(cls, root: Union[str, Path], pattern: str = "*.json", recursive: bool = True) -> "MCCCorpus":
        root = Path(root)
        paths = list(root.rglob(pattern)) if recursive else list(root.glob(pattern))
        paths = [p for p in paths if p.is_file()]
        return cls(paths)

    @classmethod
    def from_paths(cls, paths: Iterable[Union[str, Path]]) -> "MCCCorpus":
        return cls([Path(p) for p in paths])

    def summary_df(self) -> pd.DataFrame:
        return build_summary_df(self.json_paths)

    def definition_index_df(self) -> pd.DataFrame:
        return build_definition_index_df(self.json_paths)

    def get_mcc(self, json_path: Union[str, Path], definition_name: str) -> Optional[int]:
        obj = load_json(json_path)
        return definition_mcc_from_obj(obj, definition_name)

    def get_mcc_from_json_str(self, json_str: str, definition_name: str) -> Optional[int]:
        return definition_mcc_from_json_str(json_str, definition_name)

    def get_cfg_tree(
        self,
        json_path: Union[str, Path],
        definition_name: str,
        start: Union[int, str, None] = "entry",
        max_depth: Optional[int] = None,
        include_node_info: bool = True,
    ) -> Dict[str, Any]:
        obj = load_json(json_path)
        d = find_definition(obj, definition_name)
        mcc_block = d.get("mcc")
        if not mcc_block:
            return {"root": None, "tree": {}, "nodes": {} if include_node_info else None}
        return cfg_spanning_tree(mcc_block, start=start, max_depth=max_depth, include_node_info=include_node_info)

    def get_cfg_subgraph(
        self,
        json_path: Union[str, Path],
        definition_name: str,
        keep_node_ids: Set[int],
    ) -> Dict[str, Any]:
        obj = load_json(json_path)
        d = find_definition(obj, definition_name)
        mcc_block = d.get("mcc")
        if not mcc_block:
            return {"entry": None, "exit": None, "nodes": [], "edges": []}
        return cfg_subgraph(mcc_block, keep_node_ids=keep_node_ids)

    def get_definition_mcc_trees(
        self,
        json_path: Union[str, Path],
        roots: List[str],
        max_depth: Optional[int] = None,
        only_return: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        obj = load_json(json_path)
        return definition_mcc_tree(obj, roots=roots, max_depth=max_depth, only_return=only_return)

    def get_stats(
        self,
        json_path: Union[str, Path],
        definition_name: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Returns (df, summary) where df has one row per definition:
          definition_name AND all definitions prefixed with 'definition_name::'
        """
        obj = load_json(json_path)
        df = definition_group_stats_df(obj, base_name=definition_name, json_path=json_path)
        summary = summarize_definition_group(df)
        return df, summary

# -----------------------------
# CLI
# -----------------------------
def _cmd_index(args: argparse.Namespace) -> int:
    corpus = MCCCorpus.from_dir(args.input, pattern=args.pattern, recursive=not args.no_recursive)
    df = corpus.summary_df()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".csv":
            df.to_csv(out, index=False)
        elif out.suffix.lower() in {".jsonl", ".json"}:
            # JSON Lines is usually the safest for large tables
            df.to_json(out, orient="records", lines=True)
        else:
            # fallback
            df.to_csv(out.with_suffix(".csv"), index=False)

    # Print a small view to stdout
    cols = ["json_path", "total_mcc", "max_mcc", "avg_mcc", "num_declarations_with_mcc"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(args.head).to_string(index=False))
    return 0

def _cmd_mcc(args: argparse.Namespace) -> int:
    corpus = MCCCorpus.from_paths([args.json])
    score = corpus.get_mcc(args.json, args.definition)
    print(score)
    return 0

def _cmd_tree(args: argparse.Namespace) -> int:
    corpus = MCCCorpus.from_paths([args.json])
    tree = corpus.get_cfg_tree(
        args.json,
        args.definition,
        start=args.start,
        max_depth=args.depth,
        include_node_info=not args.no_node_info,
    )
    print(json.dumps(tree, indent=2))
    return 0

def _cmd_def_tree(args: argparse.Namespace) -> int:
    corpus = MCCCorpus.from_paths([args.json])
    roots = args.roots
    only = set(args.only) if args.only else None
    out = corpus.get_definition_mcc_trees(args.json, roots=roots, max_depth=args.depth, only_return=only)
    print(json.dumps(out, indent=2))
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="mcc_tools", description="MCC JSON processing utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build 1-row-per-json summary index")
    p_index.add_argument("--input", required=True, help="Directory containing JSON files")
    p_index.add_argument("--pattern", default="*.json", help="Glob pattern (default: *.json)")
    p_index.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirs")
    p_index.add_argument("--out", default=None, help="Output path (.csv or .jsonl recommended)")
    p_index.add_argument("--head", type=int, default=20, help="How many rows to print")
    p_index.set_defaults(func=_cmd_index)

    p_mcc = sub.add_parser("mcc", help="Print MCC score for one definition in one JSON file")
    p_mcc.add_argument("--json", required=True, help="Path to JSON file")
    p_mcc.add_argument("--definition", required=True, help="Definition name (exact)")
    p_mcc.set_defaults(func=_cmd_mcc)

    p_tree = sub.add_parser("tree", help="Print CFG spanning tree (BFS) for one definition")
    p_tree.add_argument("--json", required=True, help="Path to JSON file")
    p_tree.add_argument("--definition", required=True, help="Definition name (exact)")
    p_tree.add_argument("--start", default="entry", help="entry|exit|<node_id int> (default: entry)")
    p_tree.add_argument("--depth", type=int, default=None, help="Max depth")
    p_tree.add_argument("--no-node-info", action="store_true", help="Do not include node metadata in output")
    p_tree.set_defaults(func=_cmd_tree)

    p_def_tree = sub.add_parser("def-tree", help="Definition dependency tree annotated with MCC")
    p_def_tree.add_argument("--json", required=True, help="Path to JSON file")
    p_def_tree.add_argument("--roots", nargs="+", required=True, help="One or more root definitions")
    p_def_tree.add_argument("--depth", type=int, default=None, help="Max depth")
    p_def_tree.add_argument("--only", nargs="*", default=None, help="Restrict returned nodes to this set")
    p_def_tree.set_defaults(func=_cmd_def_tree)

    args = parser.parse_args(argv)

    # allow --start 12 parsing for cfg tree
    if getattr(args, "cmd", None) == "tree":
        try:
            if args.start not in ("entry", "exit", None):
                args.start = int(args.start)
        except ValueError:
            pass

    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
