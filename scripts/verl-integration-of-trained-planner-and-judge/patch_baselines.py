#!/usr/bin/env python3
"""
patch_baselines.py — automatically convert ToolHop-only baselines to
dataset-aware (ToolHop + NESTFUL) versions.

Applies the 5 edits documented in CONVERT_OTHER_BASELINES.md to:
  - lats_baseline.py
  - alpha_umi_baseline.py
  - tool_planner_baseline.py
  - gnn4taskplan_baseline.py

All patches are idempotent (safe to re-run on already-patched files).

Usage:
    # Patch each file in place (creates .bak backups):
    python patch_baselines.py lats_baseline.py alpha_umi_baseline.py \\
        tool_planner_baseline.py gnn4taskplan_baseline.py

    # Patch into a separate output directory:
    python patch_baselines.py --out-dir patched/ lats_baseline.py ...

    # Preview diffs without writing anything:
    python patch_baselines.py --dry-run lats_baseline.py ...

After patching, place `dataset_utils.py` on the import path next to the
patched baselines and run them as before. They will auto-detect ToolHop
vs NESTFUL from the parquet's `data_source` field; override with
`--dataset {toolhop,nestful}` if needed.
"""

import re
import sys
import shutil
import argparse
import difflib
import ast
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Individual patch functions. Each takes the file text, returns modified text,
# and is idempotent (a no-op when its change has already been applied).
# ─────────────────────────────────────────────────────────────────────────────

def patch_add_import(text: str) -> str:
    """Edit 1: insert `from dataset_utils import resolve_dataset, dataset_label`
    after the last top-of-file import."""
    if "from dataset_utils import" in text:
        return text

    lines = text.split("\n")
    last_import_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Stop scanning once we hit any top-level definition
        if stripped.startswith(("def ", "class ", "@")):
            break
        if stripped.startswith(("import ", "from ")):
            last_import_idx = i

    if last_import_idx == -1:
        raise RuntimeError("Could not find any top-of-file import statements")

    lines.insert(last_import_idx + 1, "")
    lines.insert(last_import_idx + 2,
                 "from dataset_utils import resolve_dataset, dataset_label")
    return "\n".join(lines)


def patch_add_argparse_dataset(text: str) -> str:
    """Edit 2: insert `parser.add_argument('--dataset', ...)` before
    `args = parser.parse_args()`."""
    if '"--dataset"' in text and '"auto", "toolhop", "nestful"' in text:
        return text

    pattern = re.compile(
        r'^(?P<indent>[ \t]*)args\s*=\s*parser\.parse_args\(\)',
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError("Could not find 'args = parser.parse_args()'")

    indent = match.group("indent")
    new_arg = (
        f'{indent}parser.add_argument("--dataset", default="auto",\n'
        f'{indent}                    choices=["auto", "toolhop", "nestful"],\n'
        f'{indent}                    help="Dataset for labels and metadata. "\n'
        f'{indent}                         "\'auto\' detects from the parquet\'s "\n'
        f'{indent}                         "data_source field.")\n'
    )
    return text[:match.start()] + new_arg + text[match.start():]


def patch_resolve_dataset_and_outputs(text: str) -> str:
    """Edit 3: insert `dataset = resolve_dataset(...)` after `parse_args()`
    and update the all_output / all_stats config dicts to include the
    resolved dataset."""
    # Insert the resolve_dataset block (idempotent)
    if "dataset = resolve_dataset(" not in text:
        pattern = re.compile(
            r'^(?P<indent>[ \t]*)args\s*=\s*parser\.parse_args\(\)[ \t]*\n',
            re.MULTILINE,
        )
        match = pattern.search(text)
        if not match:
            raise RuntimeError("Could not find 'args = parser.parse_args()'")

        indent = match.group("indent")
        insert = (
            f'\n{indent}# Resolve dataset (auto-detect from parquet '
            f'data_source, or explicit)\n'
            f'{indent}dataset = resolve_dataset(args.dataset, args.test_parquet)\n'
            f'{indent}print(f"\\nDataset: {{dataset_label(dataset)}} ({{dataset}})")\n'
        )
        text = text[:match.end()] + insert + text[match.end():]

    # Update all_output / all_stats config initialisations (idempotent because
    # the substitution target no longer matches once replaced).
    text = re.sub(
        r'(all_output\s*=\s*\{)\s*"config":\s*vars\(args\)',
        r'\1"config": {**vars(args), "resolved_dataset": dataset}',
        text,
    )
    text = re.sub(
        r'(all_stats\s*=\s*\{)\s*"config":\s*vars\(args\)',
        r'\1"config": {**vars(args), "resolved_dataset": dataset}',
        text,
    )
    return text


def _find_param_end(params: str, start: int) -> int:
    """Walk forward from `start` through any type annotation and/or default
    value, tracking bracket depth, and return the index of the next top-level
    `,` or `)` (or len(params) if neither is found).

    This is what the regex `[^,)]*?` would do *if* commas inside brackets
    didn't count. With nested generics like `Dict[int, str]` we need actual
    depth tracking.
    """
    depth = 0
    i = start
    while i < len(params):
        ch = params[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                return i  # outer ')'
            depth -= 1
        elif ch == "," and depth == 0:
            return i
        i += 1
    return len(params)


def _param_has_default(param_text: str) -> bool:
    """True iff `param_text` (one parameter's annotation + default) contains a
    top-level `=` (i.e. has a default value). Brackets-aware so `Optional[X]`
    doesn't confuse it."""
    depth = 0
    for ch in param_text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            return True
    return False


def patch_evaluate_definitions(text: str) -> str:
    """Edit 4a: insert `dataset: str,` immediately after `perfect_gt_by_qid`
    in each `def evaluate_*` signature.

    Bracket-aware so nested type annotations like `Dict[int, str]` are not
    split at their inner commas. If `perfect_gt_by_qid` itself has a default
    value (e.g. `= None`), the inserted `dataset` param is given a default of
    `"toolhop"` so we don't violate Python's "no defaultless param after a
    param with default" rule.
    """
    pattern = re.compile(
        r'(def\s+evaluate_\w+\s*\()'         # def name(
        r'(.*?)'                              # params  (DOTALL)
        r'(\)\s*(?:->\s*[\w\[\], .]+)?\s*:)', # ): or ) -> Type:
        re.DOTALL,
    )

    def replace(m):
        prefix, params, suffix = m.group(1), m.group(2), m.group(3)
        if re.search(r'\bdataset\s*:', params):
            return m.group(0)
        pgbq = re.search(r'\bperfect_gt_by_qid\b', params)
        if not pgbq:
            return m.group(0)
        insert_at = _find_param_end(params, pgbq.end())
        pgbq_tail = params[pgbq.end():insert_at]
        new_param = (', dataset: str = "toolhop"'
                     if _param_has_default(pgbq_tail)
                     else ', dataset: str')
        new_params = params[:insert_at] + new_param + params[insert_at:]
        return prefix + new_params + suffix

    return pattern.sub(replace, text)


def patch_evaluate_call_sites(text: str) -> str:
    """Edit 4b/c: insert `dataset=dataset` at each `evaluate_*(...)` call
    site (and equivalent `eval_kwargs = dict(...)` / `run_kwargs = dict(...)`
    blocks).

    Strategy: this runs AFTER `patch_evaluate_definitions`, so the def lines
    already contain `perfect_gt_by_qid, dataset: str,`. We can match any
    remaining `perfect_gt_by_qid,` (not followed by `dataset`) — those are
    necessarily call sites. Similarly handle `=perfect_gt_by_qid)` at end of
    kwarg blocks.
    """
    # (A) Match `perfect_gt_by_qid,` in call/kwarg context.
    #     Negative lookahead skips the already-patched def signature where
    #     `perfect_gt_by_qid,` is followed by ` dataset: str,`.
    text = re.sub(
        r'\bperfect_gt_by_qid\s*,(?!\s*dataset)',
        'perfect_gt_by_qid, dataset=dataset,',
        text,
    )

    # (B) Match `=perfect_gt_by_qid)` at the end of a kwarg block.
    #     Lookbehind requires the `=` form so we never hit an if-condition
    #     like `if x in perfect_gt_by_qid):`.
    text = re.sub(
        r'(=\s*perfect_gt_by_qid)(\s*\))',
        r'\1, dataset=dataset\2',
        text,
    )
    return text


def patch_result_dict_dataset(text: str) -> str:
    """Edit 4b (cont.): add `"dataset": dataset,` to each result dict by
    inserting it immediately above the `"query_id": ex["query_id"]` line."""
    if '"dataset": dataset,' in text:
        return text
    return re.sub(
        r'^(?P<indent>[ \t]+)("query_id":\s*ex\["query_id"\])',
        r'\g<indent>"dataset": dataset,\n\g<indent>\2',
        text,
        flags=re.MULTILINE,
    )


def _find_matching_paren(text: str, open_idx: int) -> int:
    """Given text[open_idx] == '(', return the index of the matching ')'.
    Returns -1 on failure. String literals (including escapes) are skipped
    so parens inside strings don't fool the counter."""
    assert text[open_idx] == '('
    depth = 1
    i = open_idx + 1
    in_str = False
    str_char = ''
    while i < len(text):
        ch = text[i]
        prev = text[i - 1] if i > 0 else ''
        if in_str:
            if ch == str_char and prev != '\\':
                in_str = False
        elif ch in ('"', "'"):
            in_str = True
            str_char = ch
        elif ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def patch_helper_result_builders(text: str) -> str:
    """Edit 4 (extension for best_of_n_selection.py): some files delegate
    result-dict construction to a non-`evaluate_*` helper (e.g. `_build_result`).
    For each such helper:
      (a) append `dataset: str` as a parameter to its signature, and
      (b) add `dataset=dataset` (kwarg, position-independent) to every call.

    Uses AST to find candidate helpers, then string-level rewriting to
    preserve formatting. Idempotent.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text  # caller will surface this; nothing to do here

    helpers = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and not node.name.startswith('evaluate_')):
            seg = ast.get_source_segment(text, node) or ''
            if '"query_id":' in seg and 'ex["query_id"]' in seg:
                helpers.append(node.name)

    if not helpers:
        return text

    for func_name in helpers:
        # ── (a) Append `dataset: str` to the signature ──────────────────────
        sig_pat = re.compile(
            rf'(def\s+{re.escape(func_name)}\s*\()'
            r'(.*?)'
            r'(\)\s*(?:->\s*[\w\[\], .]+)?\s*:)',
            re.DOTALL,
        )

        def replace_sig(m, _fn=func_name):
            prefix, params, suffix = m.group(1), m.group(2), m.group(3)
            if re.search(r'\bdataset\s*:', params):
                return m.group(0)
            stripped = params.rstrip()
            trailing = params[len(stripped):]
            if not stripped:
                new_params = 'dataset: str' + trailing
            elif stripped.endswith(','):
                new_params = stripped + ' dataset: str,' + trailing
            else:
                new_params = stripped + ', dataset: str' + trailing
            return prefix + new_params + suffix

        text = sig_pat.sub(replace_sig, text)

        # ── (b) Add `dataset=dataset` to every call site ─────────────────────
        # Walk through every occurrence; bracket-match to find the matching
        # ')' so multi-line calls and nested parens are handled correctly.
        call_pat = re.compile(rf'\b{re.escape(func_name)}\s*\(')
        out: List[str] = []
        cursor = 0
        for m in call_pat.finditer(text):
            # Skip the def line itself (avoids re-patching the signature)
            line_start = text.rfind('\n', 0, m.start()) + 1
            if text[line_start:m.start()].lstrip().startswith('def '):
                continue

            out.append(text[cursor:m.end()])
            open_idx = m.end() - 1  # index of the '('
            close_idx = _find_matching_paren(text, open_idx)
            if close_idx == -1:
                cursor = m.end()
                continue

            args_text = text[m.end():close_idx]
            if 'dataset=dataset' in args_text:
                out.append(args_text + ')')
            else:
                stripped = args_text.rstrip()
                trailing_ws = args_text[len(stripped):]
                if not stripped:
                    out.append('dataset=dataset' + ')')
                elif stripped.endswith(','):
                    out.append(stripped + ' dataset=dataset,' + trailing_ws + ')')
                else:
                    out.append(stripped + ', dataset=dataset' + trailing_ws + ')')
            cursor = close_idx + 1

        out.append(text[cursor:])
        text = ''.join(out)

    return text


def patch_compute_stats_labels(text: str) -> str:
    """Edit 5a: extend every `compute_stats(results, <label>)` call so the
    label string ends with ` — {dataset_label(dataset)}`.

    Handles three label shapes:
        compute_stats(results, "plain")
        compute_stats(results, f"with {var}")
        compute_stats(results, f"a {x} "       # implicit concat
                               f"b {y}")       # across lines

    For the third shape, the dataset suffix is appended to the LAST string
    in the chain (not the first). The last string is upgraded to an f-string
    if it isn't already.
    """
    # Group 2 captures one or more f?"…" pieces separated by whitespace
    # (including newlines). `\s` matches newlines, so multi-line implicit
    # concatenation is captured as a single match.
    pattern = re.compile(
        r'(compute_stats\(\s*[^,]+,\s*)'
        r'((?:f?"[^"]*"\s*)+)'
    )

    def replace(m):
        prefix, chain = m.group(1), m.group(2)
        if 'dataset_label' in chain:
            return m.group(0)

        # Find the closing quote of the LAST string in the chain
        i = len(chain) - 1
        while i >= 0 and chain[i].isspace():
            i -= 1
        if i < 0 or chain[i] != '"':
            return m.group(0)
        last_close = i

        # Find its matching open quote
        last_open = chain.rfind('"', 0, last_close)
        if last_open == -1:
            return m.group(0)

        is_fstring = last_open > 0 and chain[last_open - 1] == 'f'
        suffix = ' — {dataset_label(dataset)}'

        if is_fstring:
            new_chain = (chain[:last_close]
                         + suffix
                         + chain[last_close:])
        else:
            # Upgrade just the last segment to an f-string by inserting 'f'
            # immediately before its opening quote.
            new_chain = (chain[:last_open]
                         + 'f'
                         + chain[last_open:last_close]
                         + suffix
                         + chain[last_close:])

        return prefix + new_chain

    return pattern.sub(replace, text)


def patch_compute_stats_dict(text: str) -> str:
    """Edit 5b: insert `"dataset": results[0].get("dataset", "unknown"),`
    immediately below `"label": label,` inside compute_stats."""
    if '"dataset": results[0].get("dataset"' in text:
        return text
    return re.sub(
        r'^(?P<indent>[ \t]+)("label":\s+label,)',
        r'\g<indent>\2\n\g<indent>"dataset": results[0].get("dataset", "unknown"),',
        text,
        flags=re.MULTILINE,
    )


# Ordered pipeline. Defs MUST be patched before call sites because the
# call-site patch relies on the def already containing `dataset: str,`.
PIPELINE = [
    ("add import",                          patch_add_import),
    ("add --dataset arg",                   patch_add_argparse_dataset),
    ("resolve dataset + output dicts",      patch_resolve_dataset_and_outputs),
    ("def evaluate_* signatures",           patch_evaluate_definitions),
    ("evaluate_* call sites",               patch_evaluate_call_sites),
    ("helper result builders (e.g. _build_result)",
                                            patch_helper_result_builders),
    ("result dict dataset field",           patch_result_dict_dataset),
    ("compute_stats labels",                patch_compute_stats_labels),
    ("compute_stats dict field",            patch_compute_stats_dict),
]


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def patch_file(src: Path, dst: Path, dry_run: bool, no_backup: bool):
    """Apply the full pipeline. Returns (changed: bool, message: str)."""
    original = src.read_text()
    text = original

    for step_name, fn in PIPELINE:
        try:
            text = fn(text)
        except Exception as e:
            return False, f"step '{step_name}' raised: {e}"

    if text == original:
        return False, "no changes (already patched or no targets found)"

    # Sanity: make sure the result still parses as Python.
    try:
        ast.parse(text)
    except SyntaxError as e:
        return False, f"resulting file fails to parse: {e}"

    if dry_run:
        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile=f"a/{src.name}",
            tofile=f"b/{src.name}",
        ))
        sys.stdout.write(diff)
        return True, "dry-run (no file written)"

    if src == dst and not no_backup:
        bak = src.with_suffix(src.suffix + ".bak")
        shutil.copy(src, bak)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    return True, f"patched → {dst}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+",
                        help="Baseline files to patch")
    parser.add_argument("--out-dir", default=None,
                        help="Write patched copies here (default: in-place "
                             "with .bak backup)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show unified diff(s) without writing")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip .bak files when editing in place")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None

    mode = ("dry-run"            if args.dry_run else
            f"out-dir={out_dir}"  if out_dir       else
            "in-place (with .bak)")
    print(f"Patching {len(args.files)} file(s)  [{mode}]")
    print("─" * 70)

    n_ok = n_skip = n_err = 0
    for fpath in args.files:
        src = Path(fpath)
        if not src.exists():
            print(f"  ❌  {src}: not found")
            n_err += 1
            continue
        dst = (out_dir / src.name) if out_dir else src
        changed, msg = patch_file(src, dst, args.dry_run, args.no_backup)
        if changed:
            print(f"  ✓  {src.name}: {msg}")
            n_ok += 1
        else:
            print(f"  ⏭  {src.name}: {msg}")
            n_skip += 1

    print("─" * 70)
    print(f"  {n_ok} patched   {n_skip} skipped   {n_err} error(s)")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())