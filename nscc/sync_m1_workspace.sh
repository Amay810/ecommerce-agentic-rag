#!/usr/bin/env bash
# Non-destructive sync helper for NSCC M1 workspace.
# Run ON NSCC after copying this file, or invoke via Invoke-NsccSsh once tunnel is up.
set -euo pipefail
SRC="${1:?local staged sync directory}"
DST="${2:-/scratch/users/ntu/s250045/ecommerce-agentic-rag-main}"
mkdir -p "$DST" "$DST/logs" "$DST/data" "$DST/nscc" "$DST/scripts" "$DST/ecommerce_rag" "$DST/docs" "$DST/tests"
# Copy only required trees; never delete remote history.
rsync -a --ignore-existing "$SRC/nscc/" "$DST/nscc/" || cp -rn "$SRC/nscc/." "$DST/nscc/"
rsync -a "$SRC/ecommerce_rag/" "$DST/ecommerce_rag/"
rsync -a "$SRC/scripts/" "$DST/scripts/"
rsync -a "$SRC/docs/" "$DST/docs/"
rsync -a "$SRC/tests/" "$DST/tests/"
rsync -a "$SRC/data/compiled_retail_m1/" "$DST/data/compiled_retail_m1/"
echo "synced to $DST"
