"""Frozen-policy arm: answer-free masks from the fixed frozen model, replayed
through the audited native BM25 host.

Consolidates scripts/collect_friday_native_hosts.py (collection, single BM25
host) and scripts/prepare_qwen_full_pairs.py (P267 replay). The frozen model
is Qwen3-8B via the FRIDAY gateway (temperature 0, max_tokens 8192;
configs/selection.json). Each saved mask is replayed
through NativeSuppression in released order to seal the 'qwenmax' condition.

Output: results/locomo_frozen_masks.jsonl (resumable cache) and
results/locomo_frozen_pairs.json (1540 cases, conditions static/frozen/count/
qwenmax).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from src.evaluation.friday import FridayGenerator
from src.index.lexical import SparseStaticIndex
from src.policy.native_suppression import NativeSuppression
from src.policy.traffic import CORRECTION_SUFFIX, MAX_FORMAT_ATTEMPTS, parse_mask
from src.runs.common import (FROZEN_MODEL, POLICY, RESULTS, digest, load_dataset,
                             replay_references, sha, student_step, write_json)

PAIRS = RESULTS / "locomo_pairs.json"
MASKS = RESULTS / "locomo_frozen_masks.jsonl"
OUT = RESULTS / "locomo_frozen_pairs.json"


class ReplayActor:
    def __init__(self, mask, messages):
        self.mask, self.messages = mask, messages

    def probabilities(self, messages, count):
        assert messages == self.messages and count == len(self.mask)
        return np.array(self.mask, dtype=float)


def collect_masks(api, cases, observations):
    cache = {}
    if MASKS.exists():
        for line in MASKS.read_text().splitlines():
            row = json.loads(line)
            assert row["protocol"] == api.protocol and row["qid"] not in cache
            assert row["observation_hash"] == digest(observations[row["qid"]])
            assert row["mask"] == parse_mask(row["attempts"][-1]["answers"][0],
                                             len(row["feedback_ids"]))
            cache[row["qid"]] = row
    for c in cases:
        qid = c["qid"]
        if qid in cache:
            continue
        messages = observations[qid]
        attempts = []
        while True:
            request = messages if not attempts else messages + [
                dict(role="assistant", content=attempts[-1]["answers"][0]),
                dict(role="user", content=CORRECTION_SUFFIX)]
            response = api.generate([request], 8192)
            attempts.append(response)
            try:
                mask = parse_mask(response["answers"][0], len(c["feedback_ids"]))
                break
            except (ValueError, KeyError, TypeError):
                if len(attempts) >= MAX_FORMAT_ATTEMPTS:
                    write_json(RESULTS / "locomo_frozen.failed_response.json",
                               dict(qid=qid, attempts=attempts))
                    raise
        row = dict(qid=qid, group=c["group"], feedback_ids=c["feedback_ids"],
                   observation_hash=digest(messages), protocol=api.protocol,
                   mask=mask, attempts=attempts)
        with MASKS.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        cache[qid] = row
        if len(cache) % 25 == 0:
            print("FROZEN_MASKS", len(cache), "/", len(cases), flush=True)
    return cache


def main():
    source = json.loads(PAIRS.read_text())
    assert source["status"] == "prepared" and len(source["cases"]) == 1540
    cases = source["cases"]
    responses = {r["key"]: r for r in
                 map(json.loads, (RESULTS / "locomo_pairs.responses.jsonl").read_text().splitlines())}
    observations = {}
    for c in cases:
        messages = responses[c["response_key"]]["messages"]
        assert len(messages) == 2 and [m["role"] for m in messages] == ["system", "user"]
        observations[c["qid"]] = messages

    api = FridayGenerator(FROZEN_MODEL)
    masks = collect_masks(api, cases, observations)
    assert len(masks) == 1540

    items, _ = load_dataset("locomo")
    texts = {i.uid: i.content() for i in items}
    identity = dict(pairs_sha256=sha(PAIRS), masks_sha256=sha(MASKS),
                    policy={k: POLICY[k] for k in ("strength", "kappa", "pool", "prior_weight")})
    if OUT.exists():
        assert json.loads(OUT.read_text())["identity"] == identity
        print("FROZEN_PAIRS_EXIST", flush=True)
        return
    groups = list(dict.fromkeys(c["group"] for c in cases))
    result = []
    for group in groups:
        selected = [c for c in cases if c["group"] == group]
        host = SparseStaticIndex([i for i in items if i.conv_id == group])
        replay_references(host, selected)
        policy = NativeSuppression(host)
        for case in selected:
            row = masks[case["qid"]]
            assert row["feedback_ids"] == case["feedback_ids"]
            actor = ReplayActor(row["mask"], observations[case["qid"]])
            condition, mask = student_step(policy, actor, case, texts, POLICY["threshold"])
            assert mask == row["mask"]
            conditions = copy.deepcopy(case["conditions"])
            conditions["qwenmax"] = condition
            result.append(dict(qid=case["qid"], group=group, question=case["question"],
                               reference=case["reference"], conditions=conditions,
                               qwenmax_mask=mask))
        print("FROZEN_REPLAYED", group, len(selected), flush=True)
    write_json(OUT, dict(status="prepared", identity=identity, data_sha256=source["data_sha256"],
                         policy_protocol=api.protocol,
                         scope="Full1540 LoCoMo frozen-policy replay; historically exposed; not independent-test evidence",
                         cases=result))
    print("FROZEN_PAIRS_COMPLETE", len(result), flush=True)


if __name__ == "__main__":
    main()
