#!/usr/bin/env python3
"""
Shared dataset-aware utilities for ToolHop / NESTFUL baselines.

Provides:
  - detect_dataset_from_parquet(path)   : reads `data_source` column
  - resolve_dataset(arg, parquet_path)  : --dataset auto|toolhop|nestful → str
  - get_few_shot_examples(dataset)      : ReAct few-shot block per dataset
  - DATASET_LABELS                      : pretty names for stats/output

Both datasets are normalized to the same parquet schema by
prepare_verl_rl_data.py:
  - data_source: "toolhop_planner" | "nestful_planner"
  - prompt:      [{system}, {user}]
  - reward_model.ground_truth: "Step N: {{N}} = tool(...)" formatted plan
  - extra_info.data_json:      {"question": str, "tools": {name: info}}
  - extra_info.error_type:     str (same vocabulary across datasets)
  - extra_info.quality_score:  int

So the only things that need to differ between datasets are:
  1. Few-shot examples for ReAct (queries are stylistically different).
  2. Labels / output naming (cosmetic).
Everything else — parsing, structural eval, judge scoring — works as-is.
"""

from pathlib import Path
from typing import Optional


DATASET_LABELS = {
    "toolhop": "ToolHop",
    "nestful": "NESTFUL",
}

_DATA_SOURCE_TO_DATASET = {
    "toolhop_planner": "toolhop",
    "nestful_planner": "nestful",
    "nestful_ood":     "nestful",
}


# ── Dataset detection ──────────────────────────────────────────────────────

def detect_dataset_from_parquet(parquet_path: str) -> Optional[str]:
    """
    Read the `data_source` column from a parquet and map it to a dataset name.
    Returns 'toolhop' | 'nestful' | None.
    """
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path, columns=["data_source"])
        sources = table.column("data_source").to_pylist()
        if not sources:
            return None
        # Use the most common (in case of a mixed file)
        from collections import Counter
        most_common, _ = Counter(sources).most_common(1)[0]
        return _DATA_SOURCE_TO_DATASET.get(str(most_common))
    except Exception as e:
        print(f"  ⚠  Could not auto-detect dataset from parquet: {e}")
        return None


def resolve_dataset(dataset_arg: str, parquet_path: str) -> str:
    """
    Resolve the --dataset argument:
      'auto'    → detect from parquet (fall back to 'toolhop' if detection fails)
      'toolhop' → 'toolhop'
      'nestful' → 'nestful'
    """
    if dataset_arg != "auto":
        return dataset_arg
    detected = detect_dataset_from_parquet(parquet_path)
    if detected is None:
        print("  ⚠  Auto-detect failed; defaulting to 'toolhop'. "
              "Pass --dataset explicitly to override.")
        return "toolhop"
    print(f"  ✓ Auto-detected dataset: {detected}")
    return detected


# ── ReAct few-shot examples ────────────────────────────────────────────────

TOOLHOP_FEW_SHOT_EXAMPLES = """
Example 1:

Query: What is the capital of the country where the inventor of the telephone was born?

Available Tools:
- biographical_lookup(person_name: string (required), field: string (required))
- geography_lookup(location: string (required), info_type: string (required))

Thought: I need to find the inventor of the telephone first, then find their country of birth, then look up the capital.
Step 0: {{0}} = biographical_lookup(person_name='Alexander Graham Bell', field='country_of_birth')
Thought: Now I have the country. Next I need to find its capital.
Step 1: {{1}} = geography_lookup(location={{0}}, info_type='capital_city')

---

Example 2:

Query: How many days between the birth of the US president who served during World War II and the day the war ended in Europe?

Available Tools:
- historical_event_lookup(event: string (required), field: string (required))
- person_lookup(name: string (required), field: string (required))
- date_difference(start_date: string (required), end_date: string (required), unit: string)

Thought: I need to find which US president served during World War II, then get their birth date, then find the VE Day date, then compute the difference.
Step 0: {{0}} = historical_event_lookup(event='World War II US presidency', field='president_name')
Thought: Now I have the president. I need their birth date.
Step 1: {{1}} = person_lookup(name={{0}}, field='birth_date')
Thought: Now I need the date World War II ended in Europe (VE Day).
Step 2: {{2}} = historical_event_lookup(event='VE Day', field='date')
Thought: Now I can compute the difference in days.
Step 3: {{3}} = date_difference(start_date={{1}}, end_date={{2}}, unit='days')

---

Example 3:

Query: What is the population of the city where the tallest building in the country that won the most FIFA World Cups is located?

Available Tools:
- sports_lookup(sport: string (required), query: string (required), field: string (required))
- architecture_lookup(query: string (required), country: string, field: string (required))
- geography_lookup(location: string (required), info_type: string (required))

Thought: I need to find which country has won the most FIFA World Cups.
Step 0: {{0}} = sports_lookup(sport='FIFA World Cup', query='most wins', field='country')
Thought: Now I need to find the tallest building in that country.
Step 1: {{1}} = architecture_lookup(query='tallest building', country={{0}}, field='city')
Thought: Now I need the population of that city.
Step 2: {{2}} = geography_lookup(location={{1}}, info_type='population')

---
""".strip()


# NESTFUL queries are math/computation-flavored function compositions.
# Tool names are short, API-style (add, multiply, circle_area, etc.).
# Few-shot examples reflect this register, not ToolHop's fact-retrieval style.
NESTFUL_FEW_SHOT_EXAMPLES = """
Example 1:

Query: What is the result of multiplying the sum of 7 and 13 by 4?

Available Tools:
- add(a: number (required), b: number (required))
- multiply(a: number (required), b: number (required))

Thought: I need to compute the sum of 7 and 13 first, then multiply that result by 4.
Step 0: {{0}} = add(a=7, b=13)
Thought: Now I take the sum from the previous step and multiply it by 4.
Step 1: {{1}} = multiply(a={{0}}, b=4)

---

Example 2:

Query: A rectangle has length 8 and width 5. What is the perimeter of a square whose area equals the area of this rectangle?

Available Tools:
- rectangle_area(length: number (required), width: number (required))
- square_side_from_area(area: number (required))
- square_perimeter(side: number (required))

Thought: First I need the rectangle's area, then derive the side length of a square with that area, then compute that square's perimeter.
Step 0: {{0}} = rectangle_area(length=8, width=5)
Thought: Now I find the side length of a square whose area equals the rectangle's area.
Step 1: {{1}} = square_side_from_area(area={{0}})
Thought: Finally, compute the perimeter of that square.
Step 2: {{2}} = square_perimeter(side={{1}})

---

Example 3:

Query: Given the list [3, 7, 2, 8, 5], what is the absolute difference between the maximum and the average?

Available Tools:
- list_max(values: array (required))
- list_mean(values: array (required))
- subtract(a: number (required), b: number (required))
- absolute_value(x: number (required))

Thought: I need the maximum of the list, then the mean of the list, then subtract them, then take the absolute value.
Step 0: {{0}} = list_max(values=[3, 7, 2, 8, 5])
Thought: Now compute the mean of the same list.
Step 1: {{1}} = list_mean(values=[3, 7, 2, 8, 5])
Thought: Subtract the mean from the max.
Step 2: {{2}} = subtract(a={{0}}, b={{1}})
Thought: Take the absolute value of the difference.
Step 3: {{3}} = absolute_value(x={{2}})

---
""".strip()


def get_few_shot_examples(dataset: str) -> str:
    """Return the ReAct few-shot block for the given dataset."""
    if dataset == "nestful":
        return NESTFUL_FEW_SHOT_EXAMPLES
    return TOOLHOP_FEW_SHOT_EXAMPLES  # default to ToolHop


# ── Label helpers ──────────────────────────────────────────────────────────

def dataset_label(dataset: str) -> str:
    """Pretty name for stats labels: 'toolhop' → 'ToolHop'."""
    return DATASET_LABELS.get(dataset, dataset)