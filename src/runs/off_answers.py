"""Off-policy baseline answers: the fixed BM25 host with AMIE disabled.

Consolidates the static_bm25 / membench_choice_v1 paths of the legacy
experiments/exp41_grounded_baselines.py. Per-scope SparseStaticIndex, top ten,
120-word truncation, grounded-v1 prompt, resumable predictions.jsonl under
results/off_answers/<dataset>_static_bm25/.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json

from src.index.lexical import SparseStaticIndex
from src.runs.common import (ANSWER_ENDPOINT, RESULTS, answer_generate, answer_health,
                             digest, grounded_prompt, load_dataset, write_json)


def scope_of(item, dataset):
    if dataset == "groupmem":
        return item.domain
    return getattr(item, "scope", None) or item.conv_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["locomo", "groupmem", "membench"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--endpoint", default=ANSWER_ENDPOINT)
    args = ap.parse_args()
    decoding = "membench_choice_v1" if args.dataset == "membench" else None

    health = answer_health(args.endpoint)
    items, queries = load_dataset(args.dataset)
    if args.limit or args.offset:
        queries = queries[args.offset:args.offset + args.limit if args.limit else None]
    groups = defaultdict(list)
    for item in items:
        groups[scope_of(item, args.dataset)].append(item)
    query_hash = digest([(q.qid, q.question, q.answer) for q in queries])
    content_hash = digest([(i.uid, i.content()) for i in items])
    protocol = dict(generator=health, dataset=args.dataset, method="static_bm25",
                    query_hash=query_hash, content_hash=content_hash,
                    scope="native episode/haystack; shared domain for groupmem",
                    k=10, max_words_per_unit=120, short_output_tokens=128,
                    prompt_version="grounded-v1", limit=args.limit, offset=args.offset,
                    expected=len(queries),
                    locomo_categories="answerable only; adversarial excluded")
    if decoding:
        protocol.update(decoding=decoding, prompt_version="grounded-membench-json-v2",
                        choice_scoring="json.loads(answer)[choice] == ground_truth; no normalization")
    dest = RESULTS / "off_answers" / f"{args.dataset}_static_bm25"
    dest.mkdir(parents=True, exist_ok=True)
    meta = dest / "status.json"
    rows_path = dest / "predictions.jsonl"
    if meta.exists() and json.loads(meta.read_text())["protocol"] != protocol:
        raise ValueError("Protocol mismatch")
    done = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            if row["qid"] in done:
                raise ValueError("Duplicate persisted prediction")
            done[row["qid"]] = row
    if not set(done).issubset({q.qid for q in queries}):
        raise ValueError("Unexpected persisted query")
    summary = dict(protocol=protocol, status="running", answered=len(done))
    write_json(meta, summary)
    cached_scope, index, current = None, None, None

    def record(q):
        nonlocal cached_scope, index, current
        key = getattr(q, "scope", None) or q.conv_id
        if key != cached_scope:
            current = {i.uid: i for i in groups[key]}
            if not current:
                raise ValueError(f"Empty query store: {q.qid}")
            index = SparseStaticIndex(list(current.values()))
            cached_scope = key
        selected = [current[uid] for uid, _ in index.search(q.question, 10)]
        texts = {i.uid: i.content() for i in selected}
        choices = getattr(q, "choices", None) or None
        contexts, messages = grounded_prompt([i.uid for i in selected], texts, q.question, choices)
        return dict(qid=q.qid, question=q.question,
                    references=getattr(q, "answers", None) or [q.answer],
                    ground_truth=getattr(q, "ground_truth", ""),
                    category=(getattr(q, "competency", None) or getattr(q, "qtype", None)
                              or getattr(q, "category", "")),
                    context_ids=[i.uid for i in selected], context_hash=digest(contexts),
                    messages=messages, max_tokens=128)

    def flush(batch):
        result = answer_generate([r["messages"] for r in batch], batch[0]["max_tokens"],
                                 endpoint=args.endpoint, decoding=decoding)
        with rows_path.open("a") as f:
            for row, answer in zip(batch, result["answers"]):
                row["prediction"] = answer
                row["prompt_hash"] = digest(row.pop("messages"))
                if row["ground_truth"]:
                    if decoding == "membench_choice_v1":
                        value = json.loads(answer)
                        if set(value) != {"choice"} or value["choice"] not in ("A", "B", "C", "D"):
                            raise ValueError("Invalid choice schema")
                        row["decoded_choice"] = value["choice"]
                        row["choice_correct"] = value["choice"] == row["ground_truth"]
                    else:
                        row["choice_correct"] = (answer.strip().strip(".").upper()
                                                 == row["ground_truth"].strip().upper())
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[row["qid"]] = row
        summary["answered"] = len(done)
        write_json(meta, summary)
        print(args.dataset, "static_bm25", len(done), "/", len(queries), flush=True)

    batch = []
    for q in queries:
        if q.qid in done:
            continue
        row = record(q)
        if batch and (len(batch) >= args.batch_size or row["max_tokens"] != batch[0]["max_tokens"]):
            flush(batch)
            batch = []
        batch.append(row)
    if batch:
        flush(batch)
    summary["status"] = "answers_complete"
    scores = [r["choice_correct"] for r in done.values() if "choice_correct" in r]
    if scores:
        summary["strict_choice_accuracy"] = sum(scores) / len(scores)
        summary["n_choice_scored"] = len(scores)
    write_json(meta, summary)
    print("OFF_ANSWERS_COMPLETE", args.dataset, len(done), flush=True)


if __name__ == "__main__":
    main()
