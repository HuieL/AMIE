#!/usr/bin/env bash
# Fetch MemBench into dataset/membench/ and verify the payload.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="dataset/membench/Membench"
VERIFY="dataset/membench/Membench/MemData"

if [ ! -e "$VERIFY" ]; then
    mkdir -p "$(dirname "$DEST")"
    rm -rf "$DEST"
    git clone https://github.com/import-myself/Membench "$DEST"
fi
[ -e "$VERIFY" ] || { echo "FAILED: $VERIFY missing after clone" >&2; exit 1; }
echo "OK: $VERIFY"
