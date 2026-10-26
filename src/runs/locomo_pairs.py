"""Full LoCoMo native-host pairs: sealed static/frozen/count contexts plus
GPT-5.6-Sol teacher traffic masks over the unmodified native top ten.

Consolidates scripts/collect_full_native_pairs.py (P219). Current contexts are
sealed before each update, so answers can be generated later without changing
causality. The teacher observation never contains answers or references.

Output: results/locomo_pairs.json (single merged artifact; the legacy chain
wrote per-conversation files plus a merge step) and a resumable
results/locomo_pairs.responses.jsonl teacher cache.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from src.evaluation.friday import FridayGenerator
from src.index.lexical import SparseStaticIndex
from src.policy.native_suppression import NativeSuppression
from src.policy.traffic import CORRECTION_SUFFIX, MAX_FORMAT_ATTEMPTS, observation, parse_mask
from src.runs.common import (FIT_GROUPS, POLICY, RESULTS, TEACHER_MODEL, digest,
                             grounded_prompt, load_dataset, sha, write_json)

PAIRS = RESULTS / "locomo_pairs.json"
RESPONSES = RESULTS / "locomo_pairs.responses.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true",
                    help="Verify native context identity against the off answers; no teacher calls")
    ap.add_argument("--baseline", type=str, default=str(RESULTS / "off_answers/locomo_static_bm25"),
                    help="Directory of the off-policy static_bm25 answer run")
    args = ap.parse_args()

    baseline = Path(args.baseline)
    status = json.loads((baseline / "status.json").read_text())
    assert status["status"] == "answers_complete", "Run src.runs.off_answers --dataset locomo first"
    assert status["protocol"]["k"] == 10 and status["protocol"]["method"] == "static_bm25"
    predictions = [json.loads(x) for x in (baseline / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == len({r["qid"] for r in predictions}) == 1540
    predictions = {r["qid"]: r for r in predictions}

    items, queries = load_dataset("locomo")
    assert len(queries) == 1540 and set(predictions) == {q.qid for q in queries}
    assert digest([(i.uid, i.content()) for i in items]) == status["protocol"]["content_hash"]
    assert digest([(q.qid, q.question, q.answer) for q in queries]) == status["protocol"]["query_hash"]
    data_sha = sha(Path(__file__).resolve().parents[2]
                   / "dataset/locomo/locomo/data/locomo10.json")

    api = FridayGenerator(TEACHER_MODEL)
    protocol = dict(method="static_bm25", data_sha256=data_sha, policy=api.protocol,
                    strength=POLICY["strength"], kappa=POLICY["kappa"],
                    prior_weight=POLICY["prior_weight"], pool=POLICY["pool"],
                    feedback="unmodified native host top10; query first300 and content first200 characters",
                    order="released per-conversation question order; 1540 category1-4 questions",
                    timing="seal current contexts, then update for subsequent questions; no answers or references in policy input")
    groups = defaultdict(list)
    for item in items:
        groups[item.conv_id].append(item)

    cache = {}
    if RESPONSES.exists():
        for line in RESPONSES.read_text().splitlines():
            entry = json.loads(line)
            if entry["key"] in cache:
                assert entry == cache[entry["key"]]
            cache[entry["key"]] = entry

    cases = []
    audited = 0
    for group, units in groups.items():
        host = SparseStaticIndex(units)
        texts = {i.uid: i.content() for i in units}
        frozen, count = NativeSuppression(host), NativeSuppression(host)
        group_queries = [q for q in queries if q.conv_id == group]
        for q in group_queries:
            ids = [u for u, _ in host.search(q.question, 10)]
            row = predictions[q.qid]
            contexts, prompt = grounded_prompt(ids, texts, q.question)
            assert ids == row["context_ids"] and digest(contexts) == row["context_hash"]
            assert digest(prompt) == row["prompt_hash"] and row["question"] == q.question
            assert row["references"] == [str(q.answer)] and row["category"] == q.category
            assert frozen.search(q.question) == host.search(q.question, 10)
            audited += 1
        if args.audit_only:
            continue
        for q in group_queries:
            native_ids = predictions[q.qid]["context_ids"]
            conditions = {}
            for name, index in [("static", host), ("frozen", frozen), ("count", count)]:
                ids = [u for u, _ in index.search(q.question, 10)]
                contexts, prompt = grounded_prompt(ids, texts, q.question)
                conditions[name] = dict(ids=ids, context_hash=digest(contexts),
                                        messages=prompt, prompt_hash=digest(prompt))
            prompt = observation(q.question, native_ids, texts)
            key = digest(dict(policy=api.protocol, messages=prompt))
            if key not in cache:
                attempts = []
                while True:
                    request = prompt if not attempts else prompt + [
                        dict(role="assistant", content=attempts[-1]["answers"][0]),
                        dict(role="user", content=CORRECTION_SUFFIX)]
                    response = api.generate([request], 8192)
                    attempts.append(response)
                    try:
                        mask = parse_mask(response["answers"][0], len(native_ids))
                        break
                    except (ValueError, KeyError, TypeError):
                        if len(attempts) >= MAX_FORMAT_ATTEMPTS:
                            write_json(RESULTS / "locomo_pairs.failed_response.json",
                                       dict(key=key, messages=prompt, attempts=attempts))
                            raise
                entry = dict(key=key, messages=prompt, attempts=attempts, mask=mask)
                with RESPONSES.open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                cache[key] = entry
            entry = cache[key]
            assert entry["messages"] == prompt and all(r["protocol"] == api.protocol for r in entry["attempts"])
            mask = parse_mask(entry["attempts"][-1]["answers"][0], len(native_ids))
            assert mask == entry["mask"]
            frozen.observe(native_ids, mask)
            count.observe(native_ids, [1] * len(native_ids))
            cases.append(dict(qid=q.qid, group=group, question=q.question, reference=str(q.answer),
                              conditions=conditions, feedback_ids=native_ids, mask=mask,
                              response_key=key))
        print("LOCOMO_PAIRS", group, sum(1 for c in cases if c["group"] == group), flush=True)

    if args.audit_only:
        write_json(RESULTS / "locomo_pairs.audit.json",
                   dict(audited=audited, expected=1540, native_context_identity=True))
        assert audited == 1540
        print("AUDIT_COMPLETE", audited, flush=True)
        return
    assert audited == 1540 and len(cases) == 1540
    write_json(PAIRS, dict(status="prepared", protocol=protocol, data_sha256=data_sha,
                           responses_sha256=sha(RESPONSES),
                           fit_groups=list(FIT_GROUPS),
                           scope="full existing LoCoMo; historical/training exposure; not independent evidence",
                           cases=cases))
    print("PAIRS_COMPLETE", len(cases), flush=True)


if __name__ == "__main__":
    main()
