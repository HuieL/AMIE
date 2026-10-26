"""Full-1540 trained-student pairs for one fixed seed.

Consolidates scripts/prepare_selected_full_bm25_students.py (P265). Replays
the released LoCoMo order through fresh per-conversation BM25 hosts; the
trained mask is applied with the contract threshold after each question's
answer context is sealed.

Output: results/student_pairs_s<seed>.json (conditions static/frozen/count/
student). Requires results/locomo_pairs.json and results/student_s<seed>/.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from src.index.lexical import SparseStaticIndex
from src.policy.native_suppression import NativeSuppression
from src.runs.common import POLICY, RESULTS, load_dataset, replay_references, sha, student_step, write_json

CONFIG = Path(__file__).resolve().parent / "configs" / "training.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    contract = json.loads(CONFIG.read_text())
    assert args.seed in contract["training"]["seeds"]

    pairs_path = RESULTS / "locomo_pairs.json"
    source = json.loads(pairs_path.read_text())
    assert source["status"] == "prepared"
    cases = source["cases"]
    assert len(cases) == len({c["qid"] for c in cases}) == 1540

    training = RESULTS / f"student_s{args.seed}"
    status = json.loads((training / "status.json").read_text())
    assert status["status"] == "complete" and status["seed"] == args.seed

    target = RESULTS / f"student_pairs_s{args.seed}.json"
    identity = dict(pairs_sha256=sha(pairs_path), seed=args.seed,
                    policy={k: POLICY[k] for k in ("strength", "kappa", "pool", "prior_weight")})
    if target.exists():
        assert json.loads(target.read_text())["identity"] == identity
        print("STUDENT_PAIRS_EXIST", args.seed, flush=True)
        return

    if torch_device_count() != 1:
        raise ValueError("Student replay requires exactly one visible GPU")
    import torch
    from src.policy.cumulative_actor import CumulativeMaskActor
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    model = os.environ.get("AMIE_POLICY_MODEL")
    if not model:
        raise ValueError("Set AMIE_POLICY_MODEL to a local Qwen3-8B checkpoint directory")
    actor = CumulativeMaskActor(model)
    actor.load(training / "adapter")

    items, _ = load_dataset("locomo")
    texts = {i.uid: i.content() for i in items}
    groups = list(dict.fromkeys(c["group"] for c in cases))
    result = []
    agreement = total = 0
    for group in groups:
        selected = [c for c in cases if c["group"] == group]
        host = SparseStaticIndex([i for i in items if i.conv_id == group])
        replay_references(host, selected)
        student = NativeSuppression(host)
        for case in selected:
            condition, mask = student_step(student, actor, case, texts, POLICY["threshold"])
            conditions = copy.deepcopy(case["conditions"])
            conditions["student"] = condition
            agreement += sum(a == b for a, b in zip(mask, case["mask"]))
            total += len(mask)
            result.append(dict(qid=case["qid"], group=group, question=case["question"],
                               reference=case["reference"], conditions=conditions,
                               student_mask=mask))
        print("STUDENT_PREPARED", args.seed, group, flush=True)
    write_json(target, dict(status="prepared", identity=identity, seed=args.seed,
                            data_sha256=source["data_sha256"],
                            scope="Full1540 LoCoMo trained student; historically exposed; no independent-test claim",
                            cases=result,
                            mask_agreement=dict(correct=agreement, total=total)))
    print("STUDENT_PAIRS_COMPLETE", args.seed, len(result), flush=True)


def torch_device_count():
    import torch
    return torch.cuda.device_count()


if __name__ == "__main__":
    main()
