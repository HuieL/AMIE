#!/usr/bin/env bash
# Replay control studies P620-P626 (CPU, zero API).
#
# Every study first reproduces the published replay gate (evidence Recall@10:
# off 51.94 / count 53.41 on the 1,527 evidence-valid LoCoMo questions) and
# aborts if the gate fails. The exposure study additionally reads
# results/locomo_arm_deltas.json, so it is preceded by the retrieval export
# (which needs results/locomo_pairs.json; frozen/trained arms are included
# when their pair files exist).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -e results/locomo_pairs.json ]; then
    python -m src.runs.score_answers retrieval
fi

python -m src.runs.replay_controls injection
python -m src.runs.replay_controls sufficiency
python -m src.runs.replay_controls exposure
python -m src.runs.replay_controls evolvability
python -m src.runs.replay_controls ceilings
python -m src.runs.replay_controls swap
python -m src.runs.replay_controls granularity

echo "REPLAY_CONTROLS_DONE"
