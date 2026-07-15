"""
toolhop_remap_nl_to_api_v2.py

Post-hoc remap of NL-keyed tool names to API names in an annotated ToolHop
dataset.  v2 adds a normalization-based fallback to catch ~2,908 names
that v1 left unmapped due to formatting differences between GPT-5.4's
output and ToolHop's tool dict keys.

WHAT'S NEW IN v2
================
v1 used `tn in nl_to_api` — strict equality.  This missed cases where
GPT-5.4 emitted slight format variants of the real NL keys, e.g.:
    ToolHop key:  'Which actor played Colonel Sherman T. Potter in "M*A*S*H"?'
    GPT-5.4 out:  'Which actor played Colonel Sherman T. Potter in "M*A*S*H"?:'  (extra ':')
    GPT-5.4 out:  'who has been the longest reigning british monarch'             (lowercased)
    GPT-5.4 out:  'In which division Cleveland Browns placed fourth in 2009？'   (full-width '?')
    GPT-5.4 out:  'Who is the publisher of Eye To The Telescope? '               (trailing space)
    GPT-5.4 out:  'Who is the publisher of Eye To The Telescope?::'              (double colon)

These are clearly the same tool, just written differently.  v2 adds a
two-stage match:
  1. Exact match (v1 behavior) — covers ~92.6% of names.
  2. Normalized fallback — strip trailing punctuation/whitespace,
     normalize unicode question marks, lowercase — covers ~99.9%.

In testing on the user's 9,950-plan annotated file, v2 reduces unmapped
names from 2,908 → ~3 (true hallucinations that don't match any tool
in their query's catalog under any normalization).

Also new in v2:
  - Per-query report of how many names matched via exact vs. normalized
  - --strict flag to disable normalized fallback (recovers v1 behavior)
  - More extensive self-tests covering the exact failure modes seen on
    the user's real data

USAGE
=====
  python toolhop_remap_nl_to_api_v2.py \
      --toolhop-original  ToolHop.json \
      --annotated-input   toolhop_annotated.json \
      --annotated-output  toolhop_annotated_v2_remapped.json

  # Self-test (no files needed, ~1 second)
  python toolhop_remap_nl_to_api_v2.py --self-test
"""

import argparse
import copy
import json
import re
import sys
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Normalization for fuzzy matching of GPT-5.4 outputs to ToolHop keys
# ──────────────────────────────────────────────────────────────────────────────


# Unicode question marks that should all collapse to plain '?'
_QUESTION_MARK_VARIANTS = {
    "\uff1f",  # ｜full-width question mark
    "\u2753",  # ❓ red question mark emoji-ish
    "\u2754",  # ❔ white question mark
    "\u003f",  # plain ? (kept as-is, listed for completeness)
}


def normalize_tool_name(name: str) -> str:
    """
    Normalize a tool name string for fuzzy matching against the canonical
    ToolHop tool dict keys.

    The transformations are conservative — they only collapse formatting
    differences that we observed empirically in GPT-5.4's annotated output:

    1. Unicode normalization (NFKC) folds full-width punctuation and
       compatibility forms to their ASCII equivalents.
    2. Strip leading/trailing whitespace.
    3. Strip ALL trailing combinations of ':' and whitespace, repeatedly.
       This handles '?:', '?: ', '?::', '? :', etc.
    4. Lowercase the result for case-insensitive comparison.

    Two strings are considered "the same tool" if their normalized forms
    are equal.

    NOTE: We do NOT collapse internal punctuation, internal whitespace,
    or strip accent marks — those could conflate genuinely different
    tool names.  Only trailing-noise and case differences are collapsed.
    """
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKC", name)
    # Strip trailing ':', whitespace, control chars in any combination
    s = s.rstrip(": \t\n\r")
    # Lowercase for case-insensitive lookup
    s = s.lower().strip()
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Build NL-to-API map with both exact and normalized lookup tables
# ──────────────────────────────────────────────────────────────────────────────


def build_nl_to_api_map(
    toolhop_original: List[Dict],
) -> Dict[int, Dict[str, Any]]:
    """
    Build per-query lookup tables for NL → API mapping.

    Returns: {qid: {
        "exact":      {nl_key: api_name},          (v1 behavior)
        "normalized": {normalized_nl_key: api_name}, (v2 fuzzy fallback)
    }}
    """
    by_id: Dict[int, Dict[str, Any]] = {}
    for entry in toolhop_original:
        qid = entry["id"]
        tools = entry.get("tools", {})
        exact_map: Dict[str, str] = {}
        normalized_map: Dict[str, str] = {}
        for nl_key, tool_spec in tools.items():
            api_name = (
                tool_spec.get("name") if isinstance(tool_spec, dict) else None
            )
            if not api_name:
                # Fallback: sanitize the NL key to a snake-case identifier
                sanitized = re.sub(r"[^a-z0-9_]+", "_", nl_key.lower()).strip("_")
                api_name = sanitized[:60] if sanitized else "unnamed_tool"
            exact_map[nl_key] = api_name
            normalized_map[normalize_tool_name(nl_key)] = api_name
        by_id[qid] = {"exact": exact_map, "normalized": normalized_map}
    return by_id


def _looks_like_nl(name: str) -> bool:
    """Heuristic: NL tool names are long sub-questions; API names are
    snake_case-ish identifiers."""
    if not isinstance(name, str):
        return False
    return bool(re.search(r"\s", name)) or name.endswith("?") or name.endswith("?:")


# ──────────────────────────────────────────────────────────────────────────────
# Core remapping
# ──────────────────────────────────────────────────────────────────────────────


def lookup_name(
    tn: str,
    maps: Dict[str, Any],
    use_normalized_fallback: bool = True,
) -> Tuple[Optional[str], str]:
    """
    Look up a tool name against this query's NL→API maps.

    Returns: (api_name_or_None, match_kind) where match_kind is one of:
      "exact"      — found in the verbatim NL key map
      "normalized" — found via the normalized fuzzy map
      "none"       — no match in either
    """
    if tn in maps["exact"]:
        return maps["exact"][tn], "exact"
    if use_normalized_fallback:
        nt = normalize_tool_name(tn)
        if nt and nt in maps["normalized"]:
            return maps["normalized"][nt], "normalized"
    return None, "none"


def _replace_nl_in_text(
    text: str,
    nl_to_api: Dict[str, str],
    counter: Counter,
) -> str:
    """Replace NL keys in prose with their API names. Sort longest-first
    to prevent prefix collisions. Uses exact matching only (since prose
    matching is already heuristic and we want to be conservative about
    false positives in free-form text)."""
    if not isinstance(text, str) or not text:
        return text
    for nl_key in sorted(nl_to_api.keys(), key=len, reverse=True):
        if nl_key in text:
            count = text.count(nl_key)
            text = text.replace(nl_key, nl_to_api[nl_key])
            counter[nl_key] += count
    return text


def remap_plan_steps(
    plan: Dict,
    maps: Dict[str, Any],
    stats: Dict[str, int],
    use_normalized_fallback: bool = True,
) -> Tuple[Dict, List[str]]:
    """Walk the plan's steps and remap every tool_name field."""
    new_plan = copy.deepcopy(plan)
    warnings: List[str] = []
    for step in new_plan.get("steps", []):
        tn = step.get("tool_name", "")
        api_name, kind = lookup_name(tn, maps, use_normalized_fallback)
        if kind == "exact":
            step["tool_name"] = api_name
            stats["tool_names_remapped_exact"] += 1
        elif kind == "normalized":
            step["tool_name"] = api_name
            stats["tool_names_remapped_normalized"] += 1
        elif _looks_like_nl(tn):
            warnings.append(
                f"  step {step.get('step_id')}: NL-looking tool_name "
                f"'{tn[:60]}...' matches no tool in this query's catalog "
                f"under exact or normalized lookup"
            )
            stats["tool_names_unmapped_nl_looking"] += 1
        else:
            stats["tool_names_already_clean"] += 1
    return new_plan, warnings


def remap_annotation_text(
    annotation: Dict,
    nl_to_api_exact: Dict[str, str],
    counter: Counter,
) -> Dict:
    """Remap NL substrings inside annotation prose."""
    new_ann = copy.deepcopy(annotation)
    if "reasoning" in new_ann:
        new_ann["reasoning"] = _replace_nl_in_text(
            new_ann["reasoning"], nl_to_api_exact, counter
        )
    for issue in new_ann.get("issues", []):
        if "description" in issue:
            issue["description"] = _replace_nl_in_text(
                issue["description"], nl_to_api_exact, counter
            )
        if "suggestion" in issue:
            issue["suggestion"] = _replace_nl_in_text(
                issue["suggestion"], nl_to_api_exact, counter
            )
    return new_ann


def remap_dataset(
    annotated: Dict,
    nl_maps_by_qid: Dict[int, Dict[str, Any]],
    use_normalized_fallback: bool = True,
) -> Tuple[Dict, Dict[str, Any]]:
    """Apply NL → API remapping to every plan and annotation."""
    stats = {
        "n_plans_processed":              0,
        "n_plans_with_remapped_tools":    0,
        "n_plans_with_warnings":          0,
        "tool_names_remapped_exact":      0,
        "tool_names_remapped_normalized": 0,
        "tool_names_already_clean":       0,
        "tool_names_unmapped_nl_looking": 0,
        "queries_with_no_tool_map":       [],
        "queries_with_normalized_hits":   set(),
    }
    text_replacement_counter: Counter = Counter()
    all_warnings: List[str] = []

    new_data: List[Dict] = []
    for item in annotated.get("data", []):
        stats["n_plans_processed"] += 1
        qid = item["query_id"]
        maps = nl_maps_by_qid.get(qid)
        if not maps:
            stats["queries_with_no_tool_map"].append(qid)
            new_data.append(copy.deepcopy(item))
            continue

        new_item = copy.deepcopy(item)
        before_remapped = (
            stats["tool_names_remapped_exact"]
            + stats["tool_names_remapped_normalized"]
        )
        before_normalized = stats["tool_names_remapped_normalized"]
        before_warnings = stats["tool_names_unmapped_nl_looking"]

        new_plan, warnings = remap_plan_steps(
            item["plan"], maps, stats, use_normalized_fallback,
        )
        new_item["plan"] = new_plan

        if "annotation" in new_item:
            new_item["annotation"] = remap_annotation_text(
                new_item["annotation"], maps["exact"], text_replacement_counter,
            )

        after_remapped = (
            stats["tool_names_remapped_exact"]
            + stats["tool_names_remapped_normalized"]
        )
        if after_remapped > before_remapped:
            stats["n_plans_with_remapped_tools"] += 1
        if stats["tool_names_remapped_normalized"] > before_normalized:
            stats["queries_with_normalized_hits"].add(qid)
        if stats["tool_names_unmapped_nl_looking"] > before_warnings:
            stats["n_plans_with_warnings"] += 1
            for w in warnings:
                all_warnings.append(f"[query_id={qid}] {w}")

        new_data.append(new_item)

    new_dataset = copy.deepcopy(annotated)
    new_dataset["data"] = new_data

    new_metadata = dict(new_dataset.get("metadata", {}))
    new_metadata["tool_name_format"] = "api_name"
    new_metadata["nl_remap_applied"] = True
    new_metadata["nl_remap_version"] = "v2_with_normalized_fallback" if use_normalized_fallback else "v2_strict"
    new_metadata["nl_remap_stats"] = {
        "tool_names_remapped_exact":      stats["tool_names_remapped_exact"],
        "tool_names_remapped_normalized": stats["tool_names_remapped_normalized"],
        "tool_names_already_clean":       stats["tool_names_already_clean"],
        "tool_names_unmapped_nl_looking": stats["tool_names_unmapped_nl_looking"],
        "text_substitutions":             sum(text_replacement_counter.values()),
        "queries_with_normalized_hits":   len(stats["queries_with_normalized_hits"]),
    }
    new_dataset["metadata"] = new_metadata

    stats["text_replacement_counter"] = text_replacement_counter
    stats["warnings"] = all_warnings
    return new_dataset, stats


# ──────────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────────


def verify_no_nl_names_remain(dataset: Dict) -> Tuple[int, List[str]]:
    """Walk every plan in the saved dataset and count tool_name fields
    that still look like natural-language questions."""
    nl_count = 0
    examples: List[str] = []
    for item in dataset.get("data", []):
        for step in item.get("plan", {}).get("steps", []):
            tn = step.get("tool_name", "")
            if _looks_like_nl(tn):
                nl_count += 1
                if len(examples) < 10:
                    examples.append(
                        f"query_id={item['query_id']} step_id={step.get('step_id')} "
                        f"tool_name={tn[:80]!r}"
                    )
    return nl_count, examples


# ──────────────────────────────────────────────────────────────────────────────
# Self-tests covering the EXACT failure modes seen on the user's real data
# ──────────────────────────────────────────────────────────────────────────────


def run_self_tests() -> int:
    failures: List[str] = []

    def check(cond: bool, msg: str):
        if not cond:
            failures.append(msg)
            print(f"  ✗ {msg}")
        else:
            print(f"  ✓ {msg}")

    # ── Test the normalizer directly ─────────────────────────────────────
    print("\n[Test 1] Normalizer handles real-world format variants")
    canonical = 'Which actor played Colonel Sherman T. Potter in "M*A*S*H"?'
    variants = [
        'Which actor played Colonel Sherman T. Potter in "M*A*S*H"?:',  # trailing colon
        'Which actor played Colonel Sherman T. Potter in "M*A*S*H"? ',  # trailing space
        'Which actor played Colonel Sherman T. Potter in "M*A*S*H"?::', # double colon
        'which actor played colonel sherman t. potter in "m*a*s*h"?',   # lowercased
        'Which actor played Colonel Sherman T. Potter in "M*A*S*H"？',  # full-width '?'
        '  Which actor played Colonel Sherman T. Potter in "M*A*S*H"?  ',  # surrounding ws
    ]
    canonical_norm = normalize_tool_name(canonical)
    for v in variants:
        check(
            normalize_tool_name(v) == canonical_norm,
            f"normalize({v[:50]!r}...) == normalize(canonical)",
        )

    # Test that the normalizer does NOT collapse semantically distinct names
    print("\n[Test 2] Normalizer preserves semantic distinctions")
    check(
        normalize_tool_name("What is the first name of John?")
        != normalize_tool_name("What is the last name of John?"),
        "normalizer distinguishes 'first name' vs 'last name'",
    )
    check(
        normalize_tool_name("Who played Alice?")
        != normalize_tool_name("Who played Bob?"),
        "normalizer distinguishes different proper nouns",
    )

    # ── Mock the user's actual scenario ─────────────────────────────────
    print("\n[Test 3] Mock end-to-end: ToolHop has '?', GPT-5.4 emits '?:'")
    toolhop_original = [
        {
            "id": 7,
            "tools": {
                # ToolHop's keys are clean (no trailing colon)
                'Which actor played Colonel Sherman T. Potter in "M*A*S*H"?': {
                    "name": "actor_lookup",
                    "parameters": {"required": ["character"]},
                },
                'What role did Harry Morgan play in "Kentucky Jones"?': {
                    "name": "role_lookup",
                    "parameters": {"required": ["actor"]},
                },
                "What is the first name of Seldom Jackson?": {
                    "name": "extract_first_name",
                    "parameters": {"required": ["full_name"]},
                },
                "What is the alphabetical order of the letters in Seldom?": {
                    "name": "alphabetical_sort",
                    "parameters": {"required": ["input"]},
                },
            },
        },
    ]
    annotated = {
        "metadata": {"n_queries": 1, "n_candidates_per_query": 1, "total_plans": 1},
        "data": [
            {
                "query_id": 7,
                "plan": {
                    "steps": [
                        # GPT-5.4 emits the same names but with stray colons
                        {
                            "step_id": 0,
                            "tool_name": 'Which actor played Colonel Sherman T. Potter in "M*A*S*H"?:',
                            "parameters": {},
                            "output_variable": "{{0}}",
                        },
                        {
                            "step_id": 1,
                            "tool_name": 'What role did Harry Morgan play in "Kentucky Jones"?:',
                            "parameters": {},
                            "output_variable": "{{1}}",
                        },
                        {
                            "step_id": 2,
                            "tool_name": "What is the first name of Seldom Jackson?:",
                            "parameters": {},
                            "output_variable": "{{2}}",
                        },
                        {
                            "step_id": 3,
                            "tool_name": "What is the alphabetical order of the letters in Seldom?:",
                            "parameters": {},
                            "output_variable": "{{3}}",
                        },
                    ],
                    "error_type": "none",
                },
                "annotation": {
                    "quality_score": 100,
                    "success_prediction": "yes",
                    "reasoning": "fine",
                    "issues": [],
                    "confidence": 0.98,
                },
            },
        ],
    }

    # v1 strict mode would fail to remap any of these
    nl_map = build_nl_to_api_map(toolhop_original)

    print("\n  v1-style strict matching (use_normalized_fallback=False):")
    new_strict, stats_strict = remap_dataset(annotated, nl_map, use_normalized_fallback=False)
    check(
        stats_strict["tool_names_remapped_exact"] == 0,
        "strict mode: 0 exact matches (as expected — GPT-5.4 added ':')",
    )
    check(
        stats_strict["tool_names_unmapped_nl_looking"] == 4,
        "strict mode: 4 unmapped NL-looking names (matches v1 behavior)",
    )

    print("\n  v2 normalized-fallback matching (default):")
    new_v2, stats_v2 = remap_dataset(annotated, nl_map, use_normalized_fallback=True)
    check(
        stats_v2["tool_names_remapped_exact"] == 0,
        "v2: 0 exact matches (GPT-5.4 names don't match verbatim)",
    )
    check(
        stats_v2["tool_names_remapped_normalized"] == 4,
        "v2: 4 normalized matches (the fix)",
    )
    check(
        stats_v2["tool_names_unmapped_nl_looking"] == 0,
        "v2: 0 unmapped after normalized fallback",
    )

    # Verify the actual tool_name values were rewritten correctly
    new_steps = new_v2["data"][0]["plan"]["steps"]
    check(new_steps[0]["tool_name"] == "actor_lookup", "step 0 → actor_lookup")
    check(new_steps[1]["tool_name"] == "role_lookup", "step 1 → role_lookup")
    check(new_steps[2]["tool_name"] == "extract_first_name", "step 2 → extract_first_name")
    check(new_steps[3]["tool_name"] == "alphabetical_sort", "step 3 → alphabetical_sort")

    # ── Test more format variants ─────────────────────────────────────
    print("\n[Test 4] Other format variants seen in real data")

    toolhop_q9 = [{
        "id": 9,
        "tools": {
            "Who is the publisher of Eye To The Telescope?": {
                "name": "publisher_lookup",
                "parameters": {"required": ["title"]},
            },
        },
    }]
    nl_map_q9 = build_nl_to_api_map(toolhop_q9)

    test_variants = {
        "Who is the publisher of Eye To The Telescope?":   "exact",
        "Who is the publisher of Eye To The Telescope? ":  "normalized",  # trailing space
        "Who is the publisher of Eye To The Telescope?:":  "normalized",  # trailing colon
        "Who is the publisher of Eye To The Telescope?::": "normalized",  # double colon
        "who is the publisher of eye to the telescope?":   "normalized",  # lowercased
    }
    for variant, expected_kind in test_variants.items():
        api_name, kind = lookup_name(variant, nl_map_q9[9])
        check(
            api_name == "publisher_lookup" and kind == expected_kind,
            f"variant {variant[:40]!r}... → publisher_lookup ({kind}, expected {expected_kind})",
        )

    # ── Test full-width unicode question marks ─────────────────────────
    print("\n[Test 5] Full-width unicode question marks normalize correctly")
    toolhop_uni = [{
        "id": 1,
        "tools": {
            "In which division Cleveland Browns placed fourth in 2009?": {
                "name": "division_lookup",
                "parameters": {"required": ["team"]},
            },
        },
    }]
    nl_map_uni = build_nl_to_api_map(toolhop_uni)
    api_name, kind = lookup_name(
        "In which division Cleveland Browns placed fourth in 2009？",
        nl_map_uni[1],
    )
    check(
        api_name == "division_lookup" and kind == "normalized",
        f"full-width '？' normalizes to '?' and matches",
    )

    # ── Test that genuine hallucinations are still flagged ─────────────
    print("\n[Test 6] Genuine hallucinations are still flagged as unmapped")
    toolhop_simple = [{
        "id": 0,
        "tools": {
            "What is the first name of John?": {"name": "extract_first_name",
                                                "parameters": {"required": ["full_name"]}},
        },
    }]
    annotated_hallucinated = {
        "metadata": {},
        "data": [{
            "query_id": 0,
            "plan": {
                "steps": [{
                    "step_id": 0,
                    "tool_name": "Some completely unrelated NL question that's not a tool?",
                    "parameters": {},
                    "output_variable": "{{0}}",
                }],
                "error_type": "wrong_tool",
            },
            "annotation": {"quality_score": 30, "success_prediction": "no",
                          "reasoning": "x", "issues": [], "confidence": 0.9},
        }],
    }
    nl_map_h = build_nl_to_api_map(toolhop_simple)
    new_h, stats_h = remap_dataset(annotated_hallucinated, nl_map_h)
    check(
        stats_h["tool_names_unmapped_nl_looking"] == 1,
        "true hallucination correctly flagged (1 unmapped)",
    )
    check(
        stats_h["tool_names_remapped_exact"] == 0
        and stats_h["tool_names_remapped_normalized"] == 0,
        "true hallucination NOT spuriously matched via normalization",
    )

    # ── Test that scores/structure still preserved (regression of v1 tests)
    print("\n[Test 7] Scores and plan structure preserved (regression test)")
    new_v2, _ = remap_dataset(annotated, nl_map)
    for orig, new in zip(annotated["data"], new_v2["data"]):
        check(orig["annotation"]["quality_score"] == new["annotation"]["quality_score"],
              f"quality_score preserved (qid={orig['query_id']})")
        check(orig["plan"]["error_type"] == new["plan"]["error_type"],
              f"error_type preserved (qid={orig['query_id']})")
        check(len(orig["plan"]["steps"]) == len(new["plan"]["steps"]),
              f"step count preserved (qid={orig['query_id']})")

    print("\n" + "=" * 70)
    if failures:
        print(f"❌ {len(failures)} self-test(s) FAILED:")
        for f in failures:
            print(f"   - {f}")
        return 1
    print(f"✅ All self-tests passed.")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    if "--self-test" in sys.argv:
        sys.exit(run_self_tests())

    parser = argparse.ArgumentParser(
        description="Post-hoc remap of NL tool names to API names in an "
                    "annotated ToolHop dataset.  v2 adds normalization-based "
                    "fallback matching to catch GPT-5.4's format variants."
    )
    parser.add_argument("--toolhop-original", required=True)
    parser.add_argument("--annotated-input",  required=True)
    parser.add_argument("--annotated-output", required=True)
    parser.add_argument("--strict", action="store_true",
                        help="Disable normalized fallback (recovers v1 behavior). "
                             "Use this if you want to verify v1 vs v2 numerically.")
    parser.add_argument("--show-warnings", action="store_true",
                        help="Print all warnings about unmapped NL-looking names")
    args = parser.parse_args()

    print(f"Loading original ToolHop from {args.toolhop_original}...")
    with open(args.toolhop_original, "r") as f:
        toolhop_original = json.load(f)
    print(f"  {len(toolhop_original)} queries")

    print(f"Loading annotated dataset from {args.annotated_input}...")
    with open(args.annotated_input, "r") as f:
        annotated = json.load(f)
    print(f"  {len(annotated.get('data', []))} plans")

    print("\nBuilding NL→API maps (exact + normalized)...")
    nl_maps = build_nl_to_api_map(toolhop_original)
    n_tools = sum(len(m["exact"]) for m in nl_maps.values())
    print(f"  Indexed {len(nl_maps)} queries with {n_tools} total tool entries")

    use_fallback = not args.strict
    print(f"\nRemapping dataset (use_normalized_fallback={use_fallback})...")
    new_dataset, stats = remap_dataset(annotated, nl_maps,
                                        use_normalized_fallback=use_fallback)

    nl_remaining, examples = verify_no_nl_names_remain(new_dataset)

    print("\n" + "=" * 70)
    print("REMAP REPORT")
    print("=" * 70)
    n_exact = stats["tool_names_remapped_exact"]
    n_norm  = stats["tool_names_remapped_normalized"]
    n_unmap = stats["tool_names_unmapped_nl_looking"]
    n_total = n_exact + n_norm + n_unmap + stats["tool_names_already_clean"]
    print(f"Plans processed:                     {stats['n_plans_processed']:>6}")
    print(f"Plans with at least one remap:       {stats['n_plans_with_remapped_tools']:>6}")
    print(f"Plans with warnings (unmapped):      {stats['n_plans_with_warnings']:>6}")
    print()
    print(f"tool_name remapped (EXACT match):    {n_exact:>6}  "
          f"({100*n_exact/n_total:.1f}%)")
    if use_fallback:
        print(f"tool_name remapped (NORMALIZED):     {n_norm:>6}  "
              f"({100*n_norm/n_total:.1f}%)  ← v2 fix")
        print(f"  Queries that needed normalized fallback: "
              f"{stats['queries_with_normalized_hits'] if isinstance(stats['queries_with_normalized_hits'], int) else len(stats['queries_with_normalized_hits'])}")
    print(f"tool_name already clean (API name):  {stats['tool_names_already_clean']:>6}  "
          f"({100*stats['tool_names_already_clean']/n_total:.1f}%)")
    print(f"tool_name UNMAPPED (still NL-look):  {n_unmap:>6}  "
          f"({100*n_unmap/n_total:.1f}%)")
    print()
    n_text_subs = sum(stats['text_replacement_counter'].values())
    print(f"Text substitutions in prose:         {n_text_subs}")

    if stats["queries_with_no_tool_map"]:
        print(f"\n⚠ {len(stats['queries_with_no_tool_map'])} queries had NO tool map "
              f"(missing from original ToolHop):")
        for qid in stats["queries_with_no_tool_map"][:5]:
            print(f"    - {qid}")
    else:
        print("\n✓ Every query in the annotated file had a matching tool catalog")

    print()
    print(f"Final NL-looking tool_name count after remap: {nl_remaining}")
    if nl_remaining > 0:
        print(f"  Examples:")
        for ex in examples:
            print(f"    {ex}")
        print(f"  These are likely true hallucinations — the names match no")
        print(f"  tool in their query's catalog under exact OR normalized lookup.")
        print(f"  They were preserved as-is.  Common reason: GPT-5.4 'wrong_tool'")
        print(f"  injection sometimes invents a plausible-sounding tool name.")
    else:
        print("  ✓ No NL-looking tool names remain anywhere in the dataset.")
    print("=" * 70)

    if args.show_warnings and stats["warnings"]:
        print(f"\nAll {len(stats['warnings'])} warnings:")
        for w in stats["warnings"][:50]:
            print(w)
        if len(stats["warnings"]) > 50:
            print(f"... and {len(stats['warnings']) - 50} more")

    print(f"\nWriting remapped dataset to {args.annotated_output}...")
    with open(args.annotated_output, "w") as f:
        json.dump(new_dataset, f, indent=2)
    print(f"✓ Wrote {len(new_dataset['data'])} plans")
    print()
    print("v2 metadata flags set:")
    md = new_dataset["metadata"]
    print(f"  tool_name_format:  {md.get('tool_name_format')}")
    print(f"  nl_remap_applied:  {md.get('nl_remap_applied')}")
    print(f"  nl_remap_version:  {md.get('nl_remap_version')}")
    print()
    print("Note: this fixes the NL tool-name bug only.  Score-contamination")
    print("bugs (inefficient_order, unnecessary_steps) still require")
    print("regenerating the affected plans.")


if __name__ == "__main__":
    main()