#!/usr/bin/env bash
# Train the three AMIE policy students (one GPU required).
#
# Prerequisites:
#   - results/locomo_teacher_training.json (run scripts/run_main_tables.sh
#     through the teacher-export stage first)
#   - AMIE_POLICY_MODEL set to a local Qwen/Qwen3-8B checkpoint directory
#   - exactly one visible GPU (the trainer asserts torch.cuda.device_count()==1)
#
# The contract is src/runs/configs/training.json (LoRA r=8 alpha=16 on
# q_proj/v_proj, 2 epochs, AdamW lr 1e-5, class-balanced BCE, seeds fixed).
set -euo pipefail
cd "$(dirname "$0")/.."

[ -n "${AMIE_POLICY_MODEL:-}" ] || {
    echo "Set AMIE_POLICY_MODEL to a local Qwen3-8B checkpoint directory" >&2; exit 1; }
[ -e results/locomo_teacher_training.json ] || {
    echo "results/locomo_teacher_training.json missing; run scripts/run_main_tables.sh first" >&2; exit 1; }

for s in 20260908 20260909 20260910; do
    python -m src.runs.train_policy --seed "$s"
done

echo "TRAIN_POLICY_DONE"
