#!/usr/bin/env bash
# End-to-end reproduction: datasets, main tables, policy training, replay
# controls. See README.md for endpoint/model/env requirements.
set -euo pipefail
cd "$(dirname "$0")/.."

bash dataset/locomo.sh
bash dataset/groupmem.sh
bash dataset/membench.sh

bash scripts/run_main_tables.sh
bash scripts/train_policy.sh
bash scripts/run_main_tables.sh   # second pass scores the trained arms
bash scripts/run_replay_controls.sh

echo "RUN_ALL_DONE"
