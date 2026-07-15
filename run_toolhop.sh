#!/usr/bin/env bash
set -euo pipefail

python toolhop_plan_generator.py \
  --toolhop-path ToolHop.json \
  --output-path annotated_full_toolhop.json \
  --api-key "$OPENAI_API_KEY" \
  --model gpt-4.1-mini \
  --n-candidates 10





