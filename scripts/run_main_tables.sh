#!/usr/bin/env bash
# Reproduce the main tables (LoCoMo, GroupMemBench, MemBench).
#
# CPU + API stages only; the trained-student legs run automatically if all
# three trained adapters exist under results/student_s<seed>/ (see
# scripts/train_policy.sh). Requires:
#   - datasets fetched (dataset/locomo.sh, dataset/groupmem.sh, dataset/membench.sh)
#   - an OpenAI-compatible answer endpoint (AMIE_ANSWER_ENDPOINT, default
#     http://127.0.0.1:8034) serving AMIE_ANSWER_MODEL at AMIE_ANSWER_REVISION
#   - .env with FRIDAY_API_KEY / FRIDAY_BASE_URL for the teacher and frozen
#     mask collection (src/evaluation/friday.py reads it)
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="20260908 20260909 20260910"

trained_ready() {
    for s in $SEEDS; do
        [ -e "results/student_s${s}/status.json" ] || return 1
        grep -q '"status": "complete"' "results/student_s${s}/status.json" || return 1
    done
    return 0
}

echo "=== off-arm grounded answers (all three datasets) ==="
python -m src.runs.off_answers --dataset locomo
python -m src.runs.off_answers --dataset groupmem
python -m src.runs.off_answers --dataset membench

echo "=== LoCoMo teacher pairs + frozen arm ==="
python -m src.runs.locomo_pairs
python -m src.runs.teacher_export
python -m src.runs.locomo_frozen

if trained_ready; then
    echo "=== LoCoMo trained-student pairs (adapters found) ==="
    for s in $SEEDS; do
        python -m src.runs.student_pairs --seed "$s"
    done
fi

echo "=== LoCoMo answers, official F1 report, retrieval audit ==="
python -m src.runs.score_answers answers \
    --pairs results/locomo_pairs.json \
    --out results/locomo_answers.json
python -m src.runs.score_answers report \
    --pairs results/locomo_pairs.json \
    --answers results/locomo_answers.json \
    --out results/locomo_f1_report.json
python -m src.runs.score_answers retrieval

echo "=== GroupMemBench ==="
python -m src.runs.groupmem_pipeline prepare
if trained_ready; then
    for s in $SEEDS; do
        python -m src.runs.groupmem_pipeline students --seed "$s"
        python -m src.runs.groupmem_pipeline answers --seed "$s"
        python -m src.runs.groupmem_pipeline judge --seed "$s"
    done
    python -m src.runs.groupmem_pipeline summary
else
    echo "skip trained legs: results/student_s*/ incomplete (run scripts/train_policy.sh first)"
fi

echo "=== MemBench ==="
python -m src.runs.membench_pipeline prepare
python -m src.runs.membench_pipeline audit
python -m src.runs.membench_pipeline contexts
if trained_ready; then
    for s in $SEEDS; do
        [ -e "results/membench_student_masks/${s}/complete.json" ] || \
            python -m src.runs.membench_pipeline masks --seed "$s"
        python -m src.runs.membench_pipeline finals --seed "$s"
    done
fi
python -m src.runs.membench_pipeline answers
python -m src.runs.membench_pipeline scores

echo "MAIN_TABLES_DONE"
