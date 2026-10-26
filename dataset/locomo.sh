#!/usr/bin/env bash
# Fetch the LoCoMo benchmark into dataset/locomo/ and verify the payload.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="dataset/locomo/locomo"
VERIFY="dataset/locomo/locomo/data/locomo10.json"

if [ ! -e "$VERIFY" ]; then
    mkdir -p "$(dirname "$DEST")"
    rm -rf "$DEST"
    git clone https://github.com/snap-research/locomo "$DEST"
fi
[ -e "$VERIFY" ] || { echo "FAILED: $VERIFY missing after clone" >&2; exit 1; }
echo "OK: $VERIFY"
