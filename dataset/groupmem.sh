#!/usr/bin/env bash
# Fetch GroupMemBench into dataset/groupmem/ and verify the payload.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="dataset/groupmem/GroupMemBench"
VERIFY="dataset/groupmem/GroupMemBench"

if [ ! -e "$VERIFY" ]; then
    mkdir -p "$(dirname "$DEST")"
    rm -rf "$DEST"
    git clone https://github.com/UCSB-NLP-Chang/GroupMemBench "$DEST"
fi
[ -e "$VERIFY" ] || { echo "FAILED: $VERIFY missing after clone" >&2; exit 1; }
echo "OK: $VERIFY"
