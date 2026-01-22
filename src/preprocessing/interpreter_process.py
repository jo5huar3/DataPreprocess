from __future__ import annotations
import os, io, json, tempfile, pathlib
from typing import List, Optional, Tuple, Union
from pathlib import Path, PurePosixPath
import pandas as pd
# pip install cryptol
import cryptol

def write_cryptol_tempfile(
    code: str,
    host_mount_dir: str,
    source_filename: str,          # e.g. "cryptol/examples/DEStest.cry"
    prefer_name: Optional[str] = None,
) -> Tuple[pathlib.Path, str]:
    host_mount = pathlib.Path(host_mount_dir).resolve()

    # Mirror the source file’s directory under the mount
    rel_dir = pathlib.PurePosixPath(source_filename).parent  # "cryptol/examples"
    host_target_dir = host_mount / pathlib.Path(*rel_dir.parts)
    host_target_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".cry"
    prefix = ""
    if prefer_name:
        prefix = f"{pathlib.Path(prefer_name).stem}_"

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(host_target_dir),
        prefix=prefix,
        suffix=suffix,
        delete=False,
    ) as tf:
        tf.write(code)
        tf.flush()
        os.fsync(tf.fileno())
        host_path = pathlib.Path(tf.name)

    # Build the container path using POSIX separators
    container_relpath = "files/" + str(rel_dir / host_path.name).replace("\\", "/")
    return host_path, container_relpath

def load_with_cryptol_server(
    container_relpath: str,
    server_url: Optional[str] = None,
    reset_server: bool = True,
) -> dict:
    """
    Connect to Cryptol Remote API and :load the given file path.
    Returns a dict with 'load_ok' plus optional details (file_deps, errors).
    """
    kwargs = {}
    if server_url:
        kwargs["url"] = server_url
    cry = cryptol.connect(reset_server=reset_server, **kwargs)

    p = PurePosixPath(container_relpath)
    cdir = str(p.parent)     # "files/cryptol/examples"
    base = p.name            # "tmp_ABC.cry"

    result: dict = {"load_ok": False, "file": container_relpath, "error": "None"}

    try:
        # Mirrors REPL ':load <file>'
        if hasattr(cry, "add_path"):
            cry.add_path(cdir).result()
            # Now load only by basename, like ":load tmp_ABC.cry"
            load_file = cry.load_file(base).result()
        else:
            # Fallback if your cryptol client doesn't expose add_path:
            # still loads by full path, but we keep the same interface/logging.
            cry.load_file(f"{cdir}/{base}").result()

        result["load_ok"] = True

        try:
            deps = cry.file_deps(container_relpath, is_file=True).result()
            result["file_deps"] = deps["imports"]
        except Exception:
            result["file_deps"] = "error"

    except Exception as e:
        result["error"] = e

    return result


def verify_df_row_with_cryptol(
    df: pd.DataFrame,
    idx: int,
    host_mount_dir: str,
    server_url: Optional[str] = None,
) -> dict:
    """
    Take row `idx` from df, write to mounted temp file, and ask Cryptol server to load it.
    Deletes the temp file on the host after the load attempt.

    Returns a dict with paths (for logging) and server response.
    """
    row = df.iloc[idx]
    code = row["content"]
    prefer_name = row.get("filename") if "filename" in df.columns else None

    host_path, container_relpath = write_cryptol_tempfile(
        code=code,
        host_mount_dir=host_mount_dir,
        source_filename=row["filename"],
        prefer_name=prefer_name,
    )

    try:
        load_info = load_with_cryptol_server(
            container_relpath=container_relpath,
            server_url=server_url,
            reset_server=True,  # good hygiene between calls
        )
    finally:
        # Always try to delete the temp file, even if load failed
        try:
            host_path.unlink()
            deleted = True
        except FileNotFoundError:
            deleted = False
        except OSError:
            # You might want to log this in a real system
            deleted = False

    return {
        "host_path": str(host_path),
        "container_relpath": container_relpath,
        "load_info": load_info,
        "deleted": deleted,
    }

def count_real_imports(import_lines: List[str]) -> int:
    """
    Count non-empty 'import' lines (ignores blank or comment-only lines).
    """
    return sum(
        1
        for l in import_lines
        if l.lstrip().startswith("import ")
    )

def split_import_blocks(code: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a Cryptol module into (header, imports_block, body) using a
    simple heuristic:

      * header  : everything before the first 'import ' line
      * imports : consecutive import / blank lines
      * body    : everything after the import block
    """
    lines = code.splitlines()
    header: List[str] = []
    imports: List[str] = []
    body: List[str] = []

    in_imports = False
    seen_import = False

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("import "):
            in_imports = True
            seen_import = True
            imports.append(line)
        elif in_imports and (stripped == "" or stripped.startswith("--")):
            # keep blank/comment lines inside the import block
            imports.append(line)
        elif in_imports and seen_import:
            # first non-import, non-blank after imports → body
            body.append(line)
        else:
            header.append(line)

    # If we never saw an import, all lines are in header
    if not seen_import:
        return lines, [], []

    return header, imports, body

def minimize_imports(
    code: str,
    file_relpath: Union[str, Path],
    host_mount_dir: Union[str, Path],
    server_url: str,
) -> Tuple[str, int, int]:

    file_relpath = Path(file_relpath)
    host_mount_dir = Path(host_mount_dir)

    header, imports, body = split_import_blocks(code)
    n_orig = count_real_imports(imports)

    if not imports:
        return code, 0, 0

    def _loads_ok(candidate: str) -> Tuple[bool, dict]:
        host_path, container_relpath = write_cryptol_tempfile(
            code=candidate,
            host_mount_dir=str(host_mount_dir),
            source_filename=file_relpath.as_posix(),
            prefer_name=file_relpath.name,
        )
        try:
            info = load_with_cryptol_server(
                container_relpath=container_relpath,
                server_url=server_url,
                reset_server=False,
            )
            return bool(info.get("load_ok", False)), info
        finally:
            try:
                host_path.unlink()
            except (FileNotFoundError, OSError):
                pass

    # Greedy removal loop (safe with pop)
    i = 0
    while i < len(imports):
        imp_line = imports[i]
        if not imp_line.lstrip().startswith("import "):
            i += 1
            continue

        trial_imports = imports.copy()
        removed = trial_imports.pop(i)  # remove the actual current import line
        candidate_code = "\n".join(header + trial_imports + body) + "\n"

        #print(f"    [try] Removing import at line {i}: {removed!r}")
        ok, info = _loads_ok(candidate_code)

        if ok:
            #print("    [keep removed] OK without this import")
            imports = trial_imports
            code = candidate_code
            # do NOT increment i; next line shifted into position i
        else:
            #print("    [revert] Need this import")
            i += 1



    n_final = count_real_imports(imports)
    #print(f"  [imports] Done. Final imports: {n_final}")

    final_code = "\n".join(header + imports + body) + "\n"
    return final_code, n_orig, n_final

def process_sliced_df_to_df(
    df: pd.DataFrame,
    host_mount_dir: str,
    server_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Given a DataFrame with columns 'filename' (relative path of the original .cry
    file) and 'content' (Cryptol code), validate and minimise imports.

    Returns a DataFrame with only passing rows.
    """

    if server_url is None:
        server_url = os.getenv("CRYPTOL_SERVER_URL", "http://localhost:8080")

    rows = []
    first_reset = True
    total_files = 0
    n_pass = 0
    n_fail = 0
    for idx, row in df.iterrows():
        total_files += 1
        load_info = verify_df_row_with_cryptol(
            df=df,
            idx=idx,
            host_mount_dir=str(host_mount_dir),
            server_url=server_url,
        )
        li = load_info["load_info"]
        if not li.get("load_ok", False):
            n_fail += 1
            print(f"File failed to load: {load_info['container_relpath']}")

            continue

        n_pass += 1
        final_code, n_orig, n_final = minimize_imports(
            code=row["content"],
            file_relpath=row["filename"],
            host_mount_dir=host_mount_dir,
            server_url=server_url,
        )

        rows.append({
            "filename": row["filename"],
            "filetype": "cry",
            "content": final_code,
            "n_imports_original": n_orig,
            "n_imports_final": n_final,
        })
    print(f"Processed {total_files} files: {n_pass} passed, {n_fail} failed.")
    return pd.DataFrame(rows)



# Example usage:
# df = pd.read_json("your.jsonl", lines=True)
# confirm = verify_df_row_with_cryptol(
#     df, idx=0,
#     host_mount_dir="/path/host/mount",
#     server_url=os.environ.get("CRYPTOL_SERVER_URL"),
# )
# print(json.dumps(confirm, indent=2))
