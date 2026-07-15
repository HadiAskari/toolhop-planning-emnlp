#!/usr/bin/env bash
# Restore the shipped data files to the exact paths the code expects.
#
#   1. Decompresses the *.json.gz / *.jsonl.gz artifacts in place.
#   2. Checks for the external ToolHop benchmark (not redistributed here).
#
# Run once from the repository root:  bash prepare_data.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/2] Decompressing shipped data artifacts ..."
find . -name '*.gz' -print0 | while IFS= read -r -d '' gz; do
    out="${gz%.gz}"
    if [[ -f "$out" ]]; then
        echo "  skip (exists): $out"
    else
        gunzip -k "$gz" && echo "  ok: $out"
    fi
done

echo "[2/2] Checking for the external ToolHop benchmark ..."
if [[ -f ToolHop.json && -f scripts/ToolHop.json ]]; then
    echo "  ok: ToolHop.json present."
else
    cat <<'MSG'
  MISSING: ToolHop.json

  ToolHop is a third-party benchmark and is NOT redistributed in this
  repository. Download ToolHop.json from the original ToolHop release and
  place a copy at BOTH:
      ./ToolHop.json
      ./scripts/ToolHop.json
  (The NESTFUL benchmark inputs ARE included as nestful_data.jsonl.gz.)
MSG
fi

echo "Done. See README.md > 'Data & checkpoints' for what is included."
