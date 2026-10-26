"""MemBench chronological adapter: every fixed trace, retrieve-before-ingest.

Consolidates scripts/prepare_chronological_membench_full_v2.py (P407/P408),
audit_chronological_membench_full.py (P410),
prepare_chronological_membench_count.py (P411),
run_chronological_membench_full_students_v2.py (P412/P413),
prepare_membench_student_finals.py (P431), and the MemBench table scorers.

Each question's memory trace is replayed chronologically: every incoming
message (after the first) is a history query against the prefix store, and the
terminal MCQ is answered from the final store with counts accumulated from
history feedback. Repeated memory IDs keep every occurrence via `::occ{N}`
suffixes. The off arm reproduces the exact baseline prompts; the count arm is
verified by an independent arithmetic recomputation, and each trained seed is
verified by a fresh-reload exactness check.

Stages: prepare, audit, contexts, masks, finals, answers, scores.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter, defaultdict
from dataclasses import replace
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import time

from src.datasets.membench import ROOT as DATA_ROOT, load_membench
from src.index.lexical import SparseStaticIndex
from src.policy.native_suppression import NativeSuppression
from src.policy.traffic import observation as traffic_observation
from src.runs.common import (ANSWER_ENDPOINT, MCQ_SYSTEM, POLICY, RESULTS, SEEDS,
                             answer_generate, answer_health, digest, sha, write_json)

OUT = RESULTS / "membench_cases"
CONTEXTS = RESULTS / "membench_off_count_contexts"
MASK_DIR = RESULTS / "membench_student_masks"
FINALS = RESULTS / "membench_student_finals"
QUESTIONS = 23137

sha256 = lambda b: hashlib.sha256(b).hexdigest()


def occurrence_items(units):
    counts = Counter(i.uid for i in units)
    return [replace(i, uid=i.uid + "::occ" + str(n)) if counts[i.uid] > 1 else i
            for n, i in enumerate(units)]


def original_uid(uid):
    return uid.split("::occ")[0]


def load_raw_records(queries):
    raw = {}
    for perspective, qtype in dict.fromkeys((q.perspective, q.qtype) for q in queries):
        path = DATA_ROOT / perspective / f"{qtype}.json"
        for category, traces in json.loads(path.read_text()).items():
            for trace in traces or []:
                if not isinstance(trace, dict):
                    continue
                scope = f'{perspective}:{qtype}:{category}:{trace.get("tid")}'
                flow = trace.get("message_list", [])
                turns = [t for session in flow for t in session] if flow and isinstance(flow[0], list) else flow
                assert scope not in raw
                raw[scope] = [t for t in turns if isinstance(t, dict) and t.get("mid", t.get("sid")) is not None]
    return raw


def load_baseline(path):
    status = json.loads((Path(path) / "status.json").read_text())
    assert status["status"] == "answers_complete"
    rows = {r["qid"]: r for r in map(json.loads, (Path(path) / "predictions.jsonl").read_text().splitlines())}
    assert len(rows) == QUESTIONS
    return rows, status["protocol"]


def prepare_one(qid):
    q = QUERIES[qid]
    path = OUT / (sha256(qid.encode()) + ".json.gz")
    if path.exists():
        data = json.loads(gzip.decompress(path.read_bytes()))
        assert data["qid"] == qid and data["status"] == "complete_native_history_inputs"
        return qid, len(data["cases"]) - 1
    units = GROUPS[q.scope]
    records = RAW[q.scope]
    assert len(units) == len(records) and len(units) == len({i.uid for i in units})
    texts = {i.uid: i.content() for i in units}
    cases = []
    for i, (item, record) in enumerate(zip(units, records)):
        assert original_uid(item.uid) == q.scope + "::m" + str(record.get("mid", record.get("sid")))
        query = (str(record.get("user_message", record.get("user", "")) or "")
                 if q.perspective == "FirstAgent" else str(record.get("message", item.text) or ""))
        if i == 0 or not query.strip():
            continue
        host = SparseStaticIndex(units[:i])
        ids = [u for u, _ in host.search(query, 10)]
        assert item.uid not in ids
        obs = traffic_observation(query, ids, texts)
        cases.append(dict(cid=qid + "::h" + str(i), kind="history", position=i, query=query,
                          feedback_ids=ids, observation=obs, observation_hash=digest(obs)))
    host = SparseStaticIndex(units)
    candidates = {str(k): [[u, float(s)] for u, s in host.search(q.question, k)] for k in (10, 100)}
    ids = [u for u, _ in candidates["10"]]
    obs = traffic_observation(q.question, ids, texts)
    cases.append(dict(cid=qid + "::final", kind="final", position=len(units), query=q.question,
                      feedback_ids=ids, observation=obs, observation_hash=digest(obs)))
    contexts, messages = mcq_prompt(ids, texts, q)
    baseline = BASELINES[qid]
    assert [original_uid(u) for u in ids] == baseline["context_ids"]
    assert digest(contexts) == baseline["context_hash"] and digest(messages) == baseline["prompt_hash"]
    result = dict(status="complete_native_history_inputs", qid=qid, scope=q.scope,
                  perspective=q.perspective, question_type=q.qtype, cases=cases,
                  final=dict(question=q.question, choices=q.choices, reference=q.ground_truth,
                             candidates=candidates,
                             texts={u: texts[u] for u, _ in candidates["10"] + candidates["100"]},
                             gold_uids=[i.uid for i in units if original_uid(i.uid) in q.gold_uids],
                             original_gold_uids=q.gold_uids,
                             occurrence_to_original={i.uid: original_uid(i.uid) for i in units}),
                  disabled_prompt_hash=digest(messages), baseline_prompt_verified=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(result, ensure_ascii=False).encode(), compresslevel=3, mtime=0))
    tmp.replace(path)
    return qid, len(cases) - 1


def mcq_prompt(ids, texts, q):
    contexts = [" ".join(texts[u].split()[:120]) for u in ids]
    context = "\n".join(f"[{i + 1}] {v}" for i, v in enumerate(contexts)) or "(no memories retrieved)"
    question = q.question + "\n" + q.prompt_choices()
    messages = [dict(role="system", content=MCQ_SYSTEM),
                dict(role="user", content=f"Memories:\n{context}\n\nQuestion:\n{question}")]
    return contexts, messages


def mcq_prompt_from_final(ids, final):
    texts = final["texts"]
    contexts = [" ".join(texts[u].split()[:120]) for u in ids]
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(contexts)) or "(no memories retrieved)"
    question = final["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in sorted(final["choices"].items()))
    messages = [dict(role="system", content=MCQ_SYSTEM),
                dict(role="user", content=f"Memories:\n{context}\n\nQuestion:\n{question}")]
    return messages


def cmd_prepare(args):
    assert 1 <= args.workers <= 4
    global QUERIES, GROUPS, RAW, BASELINES
    OUT.mkdir(parents=True, exist_ok=True)
    items, qs = load_membench()
    assert len(qs) == QUESTIONS
    QUERIES = {q.qid: q for q in qs}
    assert len(QUERIES) == len(qs)
    GROUPS = defaultdict(list)
    for item in items:
        GROUPS[item.scope].append(item)
    for scope, units in GROUPS.items():
        GROUPS[scope] = occurrence_items(units)
    RAW = load_raw_records(qs)
    BASELINES, protocol = load_baseline(args.baseline)
    assert set(BASELINES) == set(QUERIES)
    done = {}
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        tasks = [pool.submit(prepare_one, q.qid) for q in qs]
        for task in concurrent.futures.as_completed(tasks):
            qid, n = task.result()
            done[qid] = dict(history_observations=n)
            if len(done) % 500 == 0:
                print("MEMBENCH_PREPARED", len(done), flush=True)
    write_json(OUT / "manifest.json",
               dict(status="complete_native_history_inputs", questions=QUESTIONS, rows=done))
    print("MEMBENCH_PREPARE_COMPLETE", len(done), flush=True)


def audit_one(qid):
    q = QUERIES[qid]
    path = OUT / (sha256(qid.encode()) + ".json.gz")
    data = json.loads(gzip.decompress(path.read_bytes()))
    assert data["status"] == "complete_native_history_inputs"
    old = GROUPS_RAW[q.scope]
    raw = RAW[q.scope]
    assert len(old) == len(raw)
    repeated = Counter(i.uid for i in old)
    units = [replace(item, uid=f"{item.uid}::occ{pos}") if repeated[item.uid] > 1 else item
             for pos, item in enumerate(old)]
    assert len({i.uid for i in units}) == len(units)
    original = {item.uid: old[pos].uid for pos, item in enumerate(units)}
    counts = defaultdict(int)
    cursor = 0
    for pos, (item, record) in enumerate(zip(units, raw)):
        assert old[pos].uid == q.scope + "::m" + str(record.get("mid", record.get("sid")))
        query = (str(record.get("user_message", record.get("user", "")) or "")
                 if q.perspective == "FirstAgent" else str(record.get("message", item.text) or ""))
        if pos == 0 or not query.strip():
            continue
        event = data["cases"][cursor]
        cursor += 1
        assert (event["kind"], event["position"], event["cid"], event["query"]) == (
            "history", pos, qid + "::h" + str(pos), query)
        prefix = units[:pos]
        ids = [u for u, _ in SparseStaticIndex(prefix).search(query, 10)]
        texts = {x.uid: x.content() for x in prefix}
        assert ids == event["feedback_ids"] and item.uid not in ids and len(ids) == len(set(ids))
        obs = traffic_observation(query, ids, texts)
        assert event["observation"] == obs and event["observation_hash"] == digest(obs)
        for u in ids:
            counts[u] += 1
    assert cursor == len(data["cases"]) - 1
    final = data["final"]
    host = SparseStaticIndex(units)
    expected = {str(k): [list(v) for v in host.search(q.question, k)] for k in (10, 100)}
    assert final["candidates"] == expected and final["occurrence_to_original"] == original
    assert final["original_gold_uids"] == q.gold_uids
    assert final["gold_uids"] == [i.uid for i in units if original[i.uid] in q.gold_uids]
    texts = {i.uid: i.content() for i in units}
    assert final["texts"] == {u: texts[u] for u, _ in expected["10"] + expected["100"]}
    assert (final["question"], final["choices"], final["reference"]) == (q.question, q.choices, q.ground_truth)
    ids = [u for u, _ in expected["10"]]
    baseline = BASELINES[qid]
    assert [original[u] for u in ids] == baseline["context_ids"]
    contexts, messages = mcq_prompt(ids, texts, q)
    assert digest(contexts) == baseline["context_hash"]
    assert digest(messages) == baseline["prompt_hash"] == data["disabled_prompt_hash"]
    event = data["cases"][-1]
    obs = traffic_observation(q.question, ids, texts)
    assert (event["kind"], event["position"], event["cid"], event["query"]) == (
        "final", len(units), qid + "::final", q.question)
    assert event["feedback_ids"] == ids and event["observation"] == obs
    return qid, dict(history_observations=cursor, duplicate_occurrences=len(units) - len(repeated),
                     native_prompt_verified=True, past_only_retrieval_verified=True)


def cmd_audit(args):
    """Independent reconstruction of every raw prefix, including duplicates."""
    global QUERIES, GROUPS_RAW, RAW, BASELINES
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert manifest["status"] == "complete_native_history_inputs"
    items, queries = load_membench()
    QUERIES = {q.qid: q for q in queries}
    assert set(QUERIES) == set(manifest["rows"])
    GROUPS_RAW = defaultdict(list)
    for i in items:
        GROUPS_RAW[i.scope].append(i)
    RAW = load_raw_records(queries)
    BASELINES, _ = load_baseline(args.baseline)
    done = {}
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        tasks = [pool.submit(audit_one, qid) for qid in QUERIES]
        for task in concurrent.futures.as_completed(tasks):
            qid, result = task.result()
            done[qid] = result
            if len(done) % 500 == 0:
                print("MEMBENCH_AUDITED", len(done), flush=True)
    write_json(RESULTS / "membench_history_audit.json",
               dict(status="passed_all_23137_raw_prefix_histories", rows=done,
                    history_observations=sum(r["history_observations"] for r in done.values()),
                    duplicate_occurrences=sum(r["duplicate_occurrences"] for r in done.values()),
                    boundary="Declared incoming-message retrieval adapter; verifies raw-prefix timing and unchanged baseline prompts, not policy or answer quality."))
    print("MEMBENCH_AUDIT_COMPLETE", len(done), flush=True)


def load_case(qid):
    path = OUT / (sha256(qid.encode()) + ".json.gz")
    return json.loads(gzip.decompress(path.read_bytes())), path


def raw_gold(queries):
    gold = {}
    for q in queries:
        requested = set()
        for e in q.gold_uids:
            requested.add(e)
        gold[q.qid] = requested
    return gold


def count_adjusted(final, counts):
    """Independent arithmetic recomputation of the count arm's adjusted top ten."""
    native = [u for u, _ in final["candidates"]["10"]]
    pool = {}
    for u, s in final["candidates"]["10"] + final["candidates"]["100"]:
        if u not in pool:
            pool[u] = s
        if len(pool) == 100:
            break
    if not any(counts[u] for u in native):
        return native
    return sorted(pool, key=lambda u: -pool[u] * (1 - .5 * counts[u] / (counts[u] + 3)))[:10]


def cmd_contexts(args):
    """Seal off/count terminal MCQ contexts; retrieval report over distinct originals."""
    audit = json.loads((RESULTS / "membench_history_audit.json").read_text())
    assert audit["status"] == "passed_all_23137_raw_prefix_histories"
    _, queries = load_membench()
    gold = raw_gold(queries)
    CONTEXTS.mkdir(parents=True, exist_ok=True)
    rows, excluded = [], []
    for number, q in enumerate(queries, 1):
        data, _ = load_case(q.qid)
        final = data["final"]
        counts = defaultdict(int)
        for event in data["cases"][:-1]:
            assert event["kind"] == "history" and len(event["feedback_ids"]) == len(set(event["feedback_ids"]))
            for uid in event["feedback_ids"]:
                counts[uid] += 1

        class Host:
            def search(self, query, k):
                return [(u, s) for u, s in final["candidates"][str(k)]]

        operator = NativeSuppression(Host(), strength=.5, kappa=3, pool=100)
        operator.counts.update(counts)
        native = [u for u, _ in Host().search(final["question"], 10)]
        adjusted = [u for u, _ in operator.search(final["question"], 10)]
        assert adjusted == count_adjusted(final, counts)
        original = final["occurrence_to_original"]
        qgold = gold[q.qid]
        valid = bool(qgold) and qgold <= set(original.values())
        if not valid:
            excluded.append(dict(qid=q.qid,
                                 reason="empty evidence" if not qgold else "unresolved evidence ID"))
        arms = {}
        for arm, ids in [("off", native), ("count", adjusted)]:
            assert len(ids) == len(set(ids)) <= 10
            messages = mcq_prompt_from_final(ids, final)
            h = digest(messages)
            if arm == "off":
                assert h == data["disabled_prompt_hash"]
            arms[arm] = dict(ids=ids, original_ids=[original[u] for u in ids], prompt_hash=h,
                             messages=messages,
                             recall=len({original[u] for u in ids} & qgold) / len(qgold) if valid else None)
        result = dict(qid=q.qid, reference=final["reference"], gold=sorted(qgold),
                      perspective=data["perspective"], question_type=data["question_type"], arms=arms)
        target = CONTEXTS / (sha256(q.qid.encode()) + ".json.gz")
        blob = gzip.compress(json.dumps(result, ensure_ascii=False).encode(), mtime=0)
        if target.exists():
            assert target.read_bytes() == blob
        else:
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(blob)
            tmp.replace(target)
        rows.append(dict(qid=q.qid, perspective=data["perspective"],
                         question_type=data["question_type"], gold=sorted(qgold),
                         arms={a: {k: v for k, v in item.items() if k != "messages"}
                               for a, item in arms.items()}))
        if number % 500 == 0:
            print("MEMBENCH_CONTEXTS", number, flush=True)
    validrows = [r for r in rows if r["arms"]["off"]["recall"] is not None]
    means = {arm: statistics.mean(r["arms"][arm]["recall"] for r in validrows) for arm in ("off", "count")}
    write_json(RESULTS / "membench_count_retrieval.json",
               dict(status="complete_full_off_count_saved_retrieval", questions=QUESTIONS,
                    retrieval_scope=len(validrows), means=means, excluded=excluded, rows=rows,
                    boundary="Declared chronological BM25 adapter; exact original terminal prompts and native records retained. Recall counts distinct original annotated step IDs covered by ten occurrence slots. Count contexts verified with independent arithmetic. Gold filtering affects scoring only."))
    print("MEMBENCH_COUNT_RETRIEVAL", means, flush=True)


def cmd_masks(args):
    """All history masks for one fixed trained seed; fresh-reload exactness check."""
    assert args.seed in SEEDS
    audit = json.loads((RESULTS / "membench_history_audit.json").read_text())
    assert audit["status"] == "passed_all_23137_raw_prefix_histories"
    training = RESULTS / f"student_s{args.seed}"
    status = json.loads((training / "status.json").read_text())
    assert status["status"] == "complete" and status["seed"] == args.seed
    import torch
    from src.policy.cumulative_actor import CumulativeMaskActor
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    model = os.environ.get("AMIE_POLICY_MODEL")
    if not model:
        raise ValueError("Set AMIE_POLICY_MODEL to a local Qwen3-8B checkpoint directory")
    _, queries = load_membench()
    folder = MASK_DIR / str(args.seed)
    folder.mkdir(parents=True, exist_ok=True)
    actor = CumulativeMaskActor(model, rank=8, max_length=4096)
    actor.load(training / "adapter")
    first = None
    for number, q in enumerate(queries, 1):
        cases, _ = load_case(q.qid)
        assert cases["status"] == "complete_native_history_inputs"
        target = folder / (sha256(q.qid.encode()) + ".json")
        d = json.loads(target.read_text()) if target.exists() else dict(
            status="running", seed=args.seed, qid=q.qid, rows={})
        expected = {c["cid"] for c in cases["cases"]}
        assert set(d["rows"]) <= expected
        for c in cases["cases"]:
            if c["cid"] not in d["rows"]:
                started = time.monotonic()
                probs = actor.probabilities(c["observation"], len(c["feedback_ids"]))
                d["rows"][c["cid"]] = dict(seconds=time.monotonic() - started,
                                           probabilities=probs.tolist(),
                                           mask=(probs >= .5).astype(int).tolist(),
                                           observation_hash=digest(c["observation"]))
                if len(d["rows"]) % 200 == 0:
                    write_json(target, d)
            if first is None:
                first = (c, d["rows"][c["cid"]]["probabilities"])
        d["status"] = "complete"
        write_json(target, d)
        if number % 100 == 0:
            print("MEMBENCH_MASKS", args.seed, number, flush=True)
    del actor
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    actor = CumulativeMaskActor(model, rank=8, max_length=4096)
    actor.load(training / "adapter")
    c, expected_probs = first
    assert actor.probabilities(c["observation"], len(c["feedback_ids"])).tolist() == expected_probs
    write_json(folder / "complete.json",
               dict(status="complete", seed=args.seed, questions=QUESTIONS, fresh_reload_exact=True))
    print("MEMBENCH_MASKS_COMPLETE", args.seed, flush=True)


def cmd_finals(args):
    """Seal each trained seed's terminal MCQ contexts from its audited masks."""
    assert args.seed in SEEDS
    marker = MASK_DIR / str(args.seed) / "complete.json"
    done = json.loads(marker.read_text())
    assert done["status"] == "complete" and done["fresh_reload_exact"]
    _, queries = load_membench()
    gold = raw_gold(queries)
    folder = FINALS / str(args.seed)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in queries:
        data, _ = load_case(q.qid)
        masks = json.loads((MASK_DIR / str(args.seed) / (sha256(q.qid.encode()) + ".json")).read_text())
        control = json.loads(gzip.decompress(
            (CONTEXTS / (sha256(q.qid.encode()) + ".json.gz")).read_bytes()))
        assert masks["status"] == "complete" and masks["qid"] == data["qid"] == control["qid"]
        cases = data["cases"]
        assert cases[-1]["kind"] == "final" and all(c["kind"] == "history" for c in cases[:-1])
        assert set(masks["rows"]) == {c["cid"] for c in cases}
        counts = defaultdict(int)
        for c in cases:
            row = masks["rows"][c["cid"]]
            assert row["observation_hash"] == digest(c["observation"]) == c["observation_hash"]
            probs = row["probabilities"]
            assert len(probs) == len(c["feedback_ids"]) == len(set(c["feedback_ids"]))
            assert all(0 <= p <= 1 for p in probs)
            assert row["mask"] == [int(p >= .5) for p in probs]
            # The final-query decision cannot affect its already sealed answer context.
            if c["kind"] == "history":
                for u, v in zip(c["feedback_ids"], row["mask"]):
                    counts[u] += v
        final = data["final"]
        ids = count_adjusted(final, counts)

        class Host:
            def search(self, query, k):
                return [(u, s) for u, s in final["candidates"][str(k)]]

        operator = NativeSuppression(Host(), strength=.5, kappa=3, pool=100)
        operator.counts.update(counts)
        assert ids == [u for u, _ in operator.search(final["question"], 10)]
        messages = mcq_prompt_from_final(ids, final)
        assert control["reference"] == final["reference"]
        assert control["arms"]["off"]["prompt_hash"] == data["disabled_prompt_hash"]
        qgold = set(control["gold"])
        valid = control["arms"]["off"]["recall"] is not None
        mapping = final["occurrence_to_original"]
        item = dict(qid=data["qid"], seed=args.seed, ids=ids,
                    original_ids=[mapping[u] for u in ids], messages=messages,
                    prompt_hash=digest(messages), reference=final["reference"],
                    gold=sorted(qgold),
                    recall=len({mapping[u] for u in ids} & qgold) / len(qgold) if valid else None,
                    perspective=data["perspective"], question_type=data["question_type"])
        target = folder / (sha256(q.qid.encode()) + ".json.gz")
        blob = gzip.compress(json.dumps(item, ensure_ascii=False).encode(), mtime=0)
        if target.exists():
            assert target.read_bytes() == blob
        else:
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(blob)
            tmp.replace(target)
        rows.append({k: v for k, v in item.items() if k != "messages"})
        if len(rows) % 500 == 0:
            print("MEMBENCH_FINALS", args.seed, len(rows), flush=True)
    valid = [r for r in rows if r["recall"] is not None]
    write_json(RESULTS / f"membench_student_finals_s{args.seed}.json",
               dict(status="complete_seed_final_contexts", questions=QUESTIONS, seed=args.seed,
                    rows=rows, recall=statistics.mean(r["recall"] for r in valid),
                    retrieval_scope=len(valid)))
    print("MEMBENCH_FINALS_COMPLETE", args.seed, flush=True)


def cmd_answers(args):
    """Terminal MCQ answers for off, count, and each available trained seed."""
    health = answer_health(args.endpoint)
    arms = {"off": {}, "count": {}}
    _, queries = load_membench()
    for q in queries:
        control = json.loads(gzip.decompress(
            (CONTEXTS / (sha256(q.qid.encode()) + ".json.gz")).read_bytes()))
        for arm in ("off", "count"):
            arms[arm][q.qid] = control["arms"][arm]["prompt_hash"]
    prompts = {}
    for q in queries:
        control = json.loads(gzip.decompress(
            (CONTEXTS / (sha256(q.qid.encode()) + ".json.gz")).read_bytes()))
        for arm in ("off", "count"):
            prompts[control["arms"][arm]["prompt_hash"]] = control["arms"][arm]["messages"]
    for seed in SEEDS:
        final_path = RESULTS / f"membench_student_finals_s{seed}.json"
        if not final_path.exists():
            continue
        finals = {r["qid"]: r for r in json.loads(final_path.read_text())["rows"]}
        arms[f"trained_{seed}"] = {qid: r["prompt_hash"] for qid, r in finals.items()}
        for q in queries:
            item = json.loads(gzip.decompress(
                (FINALS / str(seed) / (sha256(q.qid.encode()) + ".json.gz")).read_bytes()))
            prompts[item["prompt_hash"]] = item["messages"]
    target = RESULTS / "membench_answers.json"
    plan = dict(generator=health, max_tokens=128, decoding="membench_choice_v1")
    result = dict(status="running", plan=plan, answers={})
    if target.exists():
        result = json.loads(target.read_text())
        assert result["plan"] == plan
    write_json(target, result)
    for key, messages in prompts.items():
        if key in result["answers"]:
            continue
        generated = answer_generate([messages], 128, endpoint=args.endpoint,
                                    decoding="membench_choice_v1")
        answer = generated["answers"][0]
        value = json.loads(answer)
        assert set(value) == {"choice"} and value["choice"] in ("A", "B", "C", "D")
        result["answers"][key] = answer
        write_json(target, result)
        if len(result["answers"]) % 250 == 0:
            print("MEMBENCH_ANSWERS", len(result["answers"]), "/", len(prompts), flush=True)
    rows = {arm: [dict(qid=qid, prompt_hash=h) for qid, h in mapping.items()]
            for arm, mapping in arms.items()}
    result.update(status="answers_complete", arms={a: len(m) for a, m in arms.items()})
    write_json(target, result)
    write_json(RESULTS / "membench_answer_index.json", dict(arms=rows))
    print("MEMBENCH_ANSWERS_COMPLETE", len(result["answers"]), flush=True)


def cmd_scores(args):
    """Native choice accuracy per arm plus paired bootstrap contrasts."""
    import numpy as np
    answers = json.loads((RESULTS / "membench_answers.json").read_text())
    assert answers["status"] == "answers_complete"
    _, queries = load_membench()
    gold = {q.qid: q.ground_truth for q in queries}
    arms = {"off": {}, "count": {}}
    for q in queries:
        control = json.loads(gzip.decompress(
            (CONTEXTS / (sha256(q.qid.encode()) + ".json.gz")).read_bytes()))
        for arm in ("off", "count"):
            arms[arm][q.qid] = control["arms"][arm]["prompt_hash"]
    for seed in SEEDS:
        final_path = RESULTS / f"membench_student_finals_s{seed}.json"
        if final_path.exists():
            finals = json.loads(final_path.read_text())["rows"]
            arms[f"trained_{seed}"] = {r["qid"]: r["prompt_hash"] for r in finals}
    order = [q.qid for q in queries]
    results = {}
    for arm, mapping in arms.items():
        rows = []
        for qid in order:
            answer = answers["answers"][mapping[qid]]
            value = json.loads(answer)
            assert set(value) == {"choice"} and value["choice"] in ("A", "B", "C", "D")
            rows.append(dict(qid=qid, choice=value["choice"],
                             correct=value["choice"] == gold[qid]))
        results[arm] = dict(accuracy=sum(r["correct"] for r in rows) / len(rows), rows=rows)
    trained = [f"trained_{s}" for s in SEEDS if f"trained_{s}" in results]
    if len(trained) == 3:
        means = {}
        for i, qid in enumerate(order):
            means[qid] = statistics.mean(results[a]["rows"][i]["correct"] for a in trained)
        results["trained_mean"] = dict(
            accuracy=statistics.mean(means.values()),
            rows=[dict(qid=qid, correct=means[qid]) for qid in order])
    rng = np.random.default_rng(20260916)
    comparisons = {}
    for arm in [*trained, "trained_mean"] if len(trained) == 3 else trained:
        comparisons[arm] = {}
        for baseline in ("off", "count"):
            diffs = np.array([float(a["correct"]) - float(b["correct"]) for a, b in
                              zip(results[arm]["rows"], results[baseline]["rows"])])
            draws = rng.integers(0, len(diffs), size=(10000, len(diffs)))
            boot = diffs[draws].mean(axis=1)
            comparisons[arm][baseline] = dict(delta=float(diffs.mean()),
                                              ci95=np.quantile(boot, [.025, .975]).tolist(),
                                              draws=10000, seed=20260916)
    out = dict(status="complete", questions=QUESTIONS,
               metric='native choice accuracy; json.loads(answer)["choice"] == ground_truth',
               results={a: dict(accuracy=r["accuracy"]) for a, r in results.items()},
               comparisons=comparisons,
               bootstrap="10000 paired question bootstrap draws, seed 20260916; traces are disjoint per question")
    write_json(RESULTS / "membench_scores.json", out)
    print(json.dumps(out["results"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--baseline", default=str(RESULTS / "off_answers/membench_static_bm25"))
    p.set_defaults(fn=cmd_prepare)
    a = sub.add_parser("audit")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--baseline", default=str(RESULTS / "off_answers/membench_static_bm25"))
    a.set_defaults(fn=cmd_audit)
    c = sub.add_parser("contexts")
    c.set_defaults(fn=cmd_contexts)
    m = sub.add_parser("masks")
    m.add_argument("--seed", type=int, required=True)
    m.set_defaults(fn=cmd_masks)
    f = sub.add_parser("finals")
    f.add_argument("--seed", type=int, required=True)
    f.set_defaults(fn=cmd_finals)
    g = sub.add_parser("answers")
    g.add_argument("--endpoint", default=ANSWER_ENDPOINT)
    g.set_defaults(fn=cmd_answers)
    s = sub.add_parser("scores")
    s.set_defaults(fn=cmd_scores)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
