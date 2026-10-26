"""Generate answers and score LoCoMo pairs with the official harness.

Consolidates scripts/score_cumulative_task_pairs.py, report_full_native_pairs.py,
report_selected_full_students.py, and report_selected_retrieval.py.

Subcommands:
  answers    Fresh matched answer pass for every sealed condition prompt.
  report     Official category F1, macro means, paired bootstrap contrasts.
  retrieval  Evidence Recall@10 per arm on evidence-valid questions; also
             exports results/locomo_arm_deltas.json for the replay studies.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from src.evaluation.locomo_official import f1 as multi_f1, f1_score
from src.runs.common import (ANSWER_ENDPOINT, RESULTS, answer_generate, digest,
                             f1_contrast_bootstrap, locomo_gold_qa, locomo_world,
                             recall, sha, write_json)


def load_pairs(path):
    source = json.loads(Path(path).read_text())
    assert source["status"] == "prepared"
    return source


def cmd_answers(args):
    source = load_pairs(args.pairs)
    out = Path(args.out)
    health = answer_health_check(args)
    plan = dict(source_sha256=sha(args.pairs), generator=health, max_tokens=128, batch_size=1)
    result = dict(status="running", plan=plan, answers={}, rows=[])
    if out.exists():
        result = json.loads(out.read_text())
        assert result["plan"] == plan
    write_json(out, result)
    for case in source["cases"]:
        for c in case["conditions"].values():
            key = digest(c["messages"])
            assert key == c["prompt_hash"]
            if key in result["answers"]:
                continue
            response = answer_generate([c["messages"]], 128, endpoint=args.endpoint)
            result["answers"][key] = response["answers"][0]
            write_json(out, result)
            if len(result["answers"]) % 25 == 0:
                print("GENERATED", len(result["answers"]), flush=True)
    result["status"] = "answers_complete"
    write_json(out, result)
    print("ANSWERS_COMPLETE", len(result["answers"]), flush=True)


def answer_health_check(args):
    from src.runs.common import answer_health
    return answer_health(args.endpoint)


def score_rows(source, answers):
    gold = locomo_gold_qa()
    rows = []
    for case in source["cases"]:
        q = gold[case["qid"]]
        assert q["question"] == case["question"] and str(q["answer"]) == case["reference"]
        reference = str(q["answer"]).split(";")[0].strip() if q["category"] == 3 else str(q["answer"])
        fn = multi_f1 if q["category"] == 1 else f1_score
        scores = {}
        for condition, value in case["conditions"].items():
            key = digest(value["messages"])
            assert key == value["prompt_hash"]
            scores[condition] = float(fn(answers["answers"][key], reference))
        rows.append(dict(qid=case["qid"], group=case["group"], scores=scores))
    return rows


def cmd_report(args):
    source = load_pairs(args.pairs)
    answers_path = Path(args.answers)
    answers = json.loads(answers_path.read_text())
    assert answers["status"] == "answers_complete"
    assert answers["plan"]["source_sha256"] == sha(args.pairs)
    rows = score_rows(source, answers)
    conditions = sorted(rows[0]["scores"])
    assert "static" in conditions
    for row in rows:
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in row["scores"].values()):
            raise ValueError("Invalid official F1")
    groups = sorted({r["group"] for r in rows})
    by_group = {g: {c: float(np.mean([r["scores"][c] for r in rows if r["group"] == g]))
                    for c in conditions} for g in groups}
    macro = {c: float(np.mean([by_group[g][c] for g in groups])) for c in conditions}
    means = {c: float(np.mean([r["scores"][c] for r in rows])) for c in conditions}
    pairs = [("frozen", "static"), ("count", "static"), ("frozen", "count")]
    if "student" in conditions:
        pairs += [("student", "static"), ("student", "frozen"), ("student", "count")]
    if "qwenmax" in conditions:
        pairs += [("qwenmax", "static"), ("qwenmax", "count")]
    comparisons = {f"{l}_minus_{r}": f1_contrast_bootstrap(rows, l, r)
                   for l, r in pairs if l in conditions and r in conditions}
    gold = locomo_gold_qa()
    by_category = {str(cat): {c: float(np.mean([r["scores"][c] for r in subset]))
                              for c in conditions}
                   for cat in (1, 2, 3, 4)
                   if (subset := [r for r in rows if gold[r["qid"]]["category"] == cat])}
    out = dict(status="complete", source_sha256=sha(args.pairs),
               answers_sha256=sha(answers_path), conditions=conditions,
               question_mean=means, macro_by_conversation=macro, by_group=by_group,
               by_category=by_category, comparisons=comparisons,
               bootstrap=dict(draws=10000, seed=20260914,
                              units="F1 in [0,1]; multiply by 100 for percentage points"),
               uncertainty="10000 paired conversation-cluster bootstrap draws; exploratory unadjusted intervals; historical exposure remains")
    target = Path(args.out)
    write_json(target, out)
    print(json.dumps({"macro": macro, "question_mean": means}), flush=True)


def cmd_retrieval(args):
    """Evidence Recall@10 over saved condition IDs; no model calls."""
    world = locomo_world()
    gold = world["gold"]
    streams = [("off", "static", RESULTS / "locomo_pairs.json"),
               ("count", "count", RESULTS / "locomo_pairs.json")]
    frozen_path = RESULTS / "locomo_frozen_pairs.json"
    if frozen_path.exists():
        streams.append(("frozen", "qwenmax", frozen_path))
    trained = []
    for seed in (20260908, 20260909, 20260910):
        path = RESULTS / f"student_pairs_s{seed}.json"
        if path.exists():
            trained.append((f"trained_{seed}", "student", path))
            streams.append(trained[-1])
    records = {}
    controls = {}
    deltas = {}
    for arm, condition, path in streams:
        prepared = load_pairs(path)
        rows = []
        for c in prepared["cases"]:
            qid = c["qid"]
            pair = {label: c["conditions"][key]["ids"]
                    for label, key in [("off", "static"), ("count", "count"), (arm, condition)]}
            for ids in pair.values():
                assert len(ids) == len(set(ids)) == 10
            if qid not in controls:
                controls[qid] = {l: pair[l] for l in ("off", "count")}
            else:
                assert controls[qid] == {l: pair[l] for l in ("off", "count")}
            if qid not in gold:
                continue
            scores = {label: recall(ids, gold[qid]) for label, ids in pair.items()}
            rows.append(dict(qid=qid, group=c["group"], recall=scores[arm]))
            deltas.setdefault(arm, {})[qid] = scores[arm] - scores["off"]
            deltas.setdefault("off", {})[qid] = 0.0
            deltas.setdefault("count", {})[qid] = scores["count"] - scores["off"]
        records[arm] = rows
    if trained:
        order = [r["qid"] for r in records["off"]]
        records["trained"] = [dict(qid=qid, group=records["off"][i]["group"],
                                   recall=statistics.mean(records[f"trained_{s}"][i]["recall"]
                                                          for s in (20260908, 20260909, 20260910)))
                              for i, qid in enumerate(order)]
        deltas["trained"] = {qid: statistics.mean(deltas[f"trained_{s}"][qid]
                                                  for s in (20260908, 20260909, 20260910))
                             for qid in order}
    means = {arm: statistics.mean(r["recall"] for r in rows) for arm, rows in records.items()}
    report = dict(status="complete_saved_retrieval_audit", benchmark="LoCoMo", host="BM25",
                  metric="Question-mean evidence Recall@10 over distinct annotated gold turn IDs",
                  answer_scope=1540, retrieval_scope=len(gold),
                  mean_recall=means,
                  delta_off={arm: value - means["off"] for arm, value in means.items()})
    write_json(RESULTS / "locomo_retrieval_report.json", report)
    write_json(RESULTS / "locomo_arm_deltas.json",
               dict(status="complete", scope="evidence-valid questions; recall deltas against off",
                    questions=sorted(gold), arms={a: deltas[a] for a in sorted(deltas)}))
    print(json.dumps(dict(mean_recall=means, delta_off=report["delta_off"])), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("answers")
    a.add_argument("--pairs", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--endpoint", default=ANSWER_ENDPOINT)
    a.set_defaults(fn=cmd_answers)
    r = sub.add_parser("report")
    r.add_argument("--pairs", required=True)
    r.add_argument("--answers", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_report)
    t = sub.add_parser("retrieval")
    t.set_defaults(fn=cmd_retrieval)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
