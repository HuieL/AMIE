"""Export the six fixed training conversations with exact teacher inputs.

Consolidates scripts/export_native_teacher_training.py (P227). Only the exact
traffic observation and the target mask enter the student artifact: no
answers, references, evidence IDs, or evaluation records.

Output: results/locomo_teacher_training.json (902 rows).
"""
from __future__ import annotations

import json

from src.datasets.locomo import load_locomo
from src.policy.traffic import observation, parse_mask
from src.runs.common import FIT_GROUPS, RESULTS, digest, sha, write_json

PAIRS = RESULTS / "locomo_pairs.json"
RESPONSES = RESULTS / "locomo_pairs.responses.jsonl"
OUT = RESULTS / "locomo_teacher_training.json"


def training_row(case, response, policy, texts):
    if case["group"] not in FIT_GROUPS or not case["qid"].startswith(case["group"] + "::"):
        raise ValueError("Non-training conversation")
    ids, mask = case["feedback_ids"], case["mask"]
    if (not 0 < len(ids) <= 10 or len(set(ids)) != len(ids) or len(mask) != len(ids)
            or any(type(v) is not int or v not in (0, 1) for v in mask)):
        raise ValueError("Invalid native feedback or mask")
    expected = observation(case["question"], ids, texts)
    if response["messages"] != expected or response["mask"] != mask:
        raise ValueError("Teacher input/mask mismatch")
    if response["key"] != case["response_key"] or response["key"] != digest(dict(policy=policy, messages=expected)):
        raise ValueError("Teacher response provenance mismatch")
    return dict(qid=case["qid"], group=case["group"], messages=expected, mask=mask,
                response_key=response["key"])


def main():
    source = json.loads(PAIRS.read_text())
    assert source["status"] == "prepared"
    protocol = source["protocol"]
    items, queries = load_locomo()
    texts = {i.uid: i.content() for i in items}
    expected = {g: [q.qid for q in queries if q.conv_id == g and not q.is_adversarial] for g in FIT_GROUPS}
    questions = {q.qid: q.question for q in queries if q.conv_id in FIT_GROUPS}
    responses = {}
    for line in RESPONSES.read_text().splitlines():
        r = json.loads(line)
        if r["key"] in responses:
            assert r == responses[r["key"]]
        responses[r["key"]] = r
    rows = []
    for group in FIT_GROUPS:
        group_cases = [c for c in source["cases"] if c["group"] == group]
        assert [c["qid"] for c in group_cases] == expected[group]
        for c in group_cases:
            assert c["question"] == questions[c["qid"]]
            r = responses[c["response_key"]]
            assert all(a["protocol"] == protocol["policy"] for a in r["attempts"])
            assert parse_mask(r["attempts"][-1]["answers"][0], len(c["feedback_ids"])) == c["mask"]
            rows.append(training_row(c, r, protocol["policy"], texts))
    assert len(rows) == len({r["qid"] for r in rows}) == 902
    out = dict(status="complete",
               scope="Only six fixed training conversations; no answers, references, evidence IDs or evaluation records in rows",
               fit_groups=list(FIT_GROUPS), method="static_bm25", parent_protocol=protocol,
               pairs_sha256=sha(PAIRS), responses_sha256=sha(RESPONSES),
               policy_interface="Exact native-host feedback; original query/content observation; ten-slot binary mask",
               rows=rows)
    if OUT.exists():
        assert json.loads(OUT.read_text()) == out
    else:
        write_json(OUT, out)
    print(json.dumps(dict(status="complete", rows=len(rows), sha256=sha(OUT))), flush=True)


if __name__ == "__main__":
    main()
