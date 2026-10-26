"""Fixed three-seed class-balanced BCE student on exact teacher observations.

Consolidates scripts/train_balanced_native_distillation.py (P250). The
contract is src/runs/configs/training.json; the backbone path comes from the
AMIE_POLICY_MODEL environment variable. Requires exactly one visible GPU.

Output: results/student_s<seed>/ with adapter/ and status.json.
"""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import time

import numpy as np
import torch

from src.policy.balanced_distillation import class_weights, balanced_bce
from src.policy.cumulative_actor import CumulativeMaskActor
from src.policy.distillation_checkpoint import save_checkpoint, restore_checkpoint
from src.runs.common import RESULTS, write_json

CONFIG = Path(__file__).resolve().parent / "configs" / "training.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    contract = json.loads(CONFIG.read_text())
    cfg = contract["training"]
    assert args.seed in cfg["seeds"] and cfg["epochs"] == 2 and cfg["expected_questions"] == 902
    model = os.environ.get("AMIE_POLICY_MODEL")
    if not model:
        raise ValueError("Set AMIE_POLICY_MODEL to a local Qwen3-8B checkpoint directory")
    if torch.cuda.device_count() != 1:
        raise ValueError("Training requires exactly one visible GPU")

    teacher_path = RESULTS / "locomo_teacher_training.json"
    teacher = json.loads(teacher_path.read_text())
    assert teacher["status"] == "complete" and teacher["fit_groups"] == list(cfg["fit_groups"])
    rows = teacher["rows"]
    assert len(rows) == len({r["qid"] for r in rows}) == 902
    for row in rows:
        assert set(row) == {"qid", "group", "messages", "mask", "response_key"}
        assert row["group"] in cfg["fit_groups"]
        assert 0 < len(row["mask"]) <= 10 and all(type(v) is int and v in (0, 1) for v in row["mask"])
    weights = class_weights(rows, cfg["fit_groups"])
    declared = cfg["class_weighting"]
    assert weights["keep_slots"] == declared["keep_slots"]
    assert weights["suppress_slots"] == declared["suppress_slots"]
    assert weights["keep_weight"] == declared["keep_weight"]
    assert weights["suppress_weight"] == declared["suppress_weight"]

    folder = RESULTS / f"student_s{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    status = folder / "status.json"
    checkpoint = folder / "training.pt"
    if status.exists():
        old = json.loads(status.read_text())
        if old["status"] == "complete" and old["seed"] == args.seed:
            print("TRAINING_COMPLETE", args.seed, flush=True)
            return

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    actor = CumulativeMaskActor(model, rank=cfg["lora_rank"], max_length=cfg["max_context_tokens"])
    actor.initialize_mask_head()
    actor.model.eval()
    peft = actor.model.peft_config["default"]
    assert peft.lora_alpha == cfg["lora_alpha"] and peft.lora_dropout == cfg["dropout"]
    assert set(peft.target_modules) == set(cfg["lora_targets"])
    parameters = [p for p in actor.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(args.seed)
    order = np.concatenate([rng.permutation(len(rows)) for _ in range(cfg["epochs"])]).tolist()
    identity = dict(seed=args.seed, contract="src/runs/configs/training.json")
    start, losses = restore_checkpoint(checkpoint, actor.model, optimizer, identity) if checkpoint.exists() else (0, [])
    assert len(losses) == start and 0 <= start <= len(order)
    record = dict(status="training", seed=args.seed, step=start, total_steps=len(order))
    write_json(status, record)
    for step in range(start, len(order)):
        row = rows[order[step]]
        mask = row["mask"]
        logits = actor.policy_logits([row["messages"]])[0, :len(mask)]
        target = torch.tensor(mask, device=logits.device, dtype=logits.dtype)
        loss = balanced_bce(logits, target, weights)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite distillation loss")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip"])
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0 or step + 1 == len(order):
            save_checkpoint(checkpoint, actor.model, optimizer, step + 1, identity, losses)
            record.update(step=step + 1, heartbeat=time.time(),
                          mean_weighted_bce=float(np.mean(losses)))
            write_json(status, record)
            print("TRAIN", args.seed, step + 1, "/", len(order),
                  "WEIGHTED_BCE", record["mean_weighted_bce"], flush=True)
    actor.save(folder / "adapter")
    record.update(status="complete",
                  epoch_mean_weighted_bce=[float(np.mean(losses[e * 902:(e + 1) * 902])) for e in range(2)])
    write_json(status, record)
    print("TRAINING_COMPLETE", args.seed, flush=True)


if __name__ == "__main__":
    main()
