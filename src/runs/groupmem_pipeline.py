"""GroupMem cross-dataset transfer: fixed LoCoMo-trained policies on BM25 hosts.

Consolidates scripts/prepare_groupmem_selected_transfer.py (P286),
prepare_groupmem_selected_students.py (P288),
generate_groupmem_selected_answers.py (P290), score_groupmem_selected_glm.py /
groupmem_judge_contract.py (P293), groupmem_paired_summary.py and
native_pair_review_helpers.py (paired domain summaries).

Stages:
  prepare   Audited native replay: per-domain hosts, sealed static condition,
            answer-free observations. Zero policy calls.
  students  One fixed seed's trained masks -> conditions static/count/student.
  answers   Matched fresh Qwen3-32B answer pass (max_tokens 128).
  judge     Native GroupMem judge prompt/parser, GPT-5 judge.
  summary   Paired per-domain effects; three seeds averaged within question.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time
import typing
import urllib.error
import urllib.request

from src.evaluation.friday import NoRedirect, friday_credentials
from src.index.lexical import SparseStaticIndex
from src.policy.native_suppression import NativeSuppression
from src.policy.traffic import observation as traffic_observation
from src.runs.common import (ANSWER_ENDPOINT, POLICY, RESULTS, SEEDS, answer_generate,
                             answer_health, digest, grounded_prompt, load_dataset,
                             sha, student_step, write_json)

OUT = RESULTS / "groupmem"
PREPARED = OUT / "prepared.json"
OBSERVATIONS = OUT / "observations.json"
JUDGE_MODEL = "GPT-5"
DOMAINS = ("Technology", "Finance", "Healthcare", "Manufacturing")


def native_functions():
    """Exact upstream judge helpers, exec'd from the cloned GroupMemBench."""
    root = Path(__file__).resolve().parents[2] / "dataset/groupmem/GroupMemBench"
    files = {"baselines/rag_common/eval_lib.py": {"extract_final", "split_reasoning_and_final", "parse_judgment"},
             "llm_utils.py": {"model_uses_max_completion_tokens", "build_chat_completion_kwargs"}}
    ns = dict(vars(typing))
    hashes = {}
    for filename, names in files.items():
        path = root / filename
        hashes[filename] = sha(path)
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns, hashes


def cmd_prepare(args):
    baseline = Path(args.baseline)
    status = json.loads((baseline / "status.json").read_text())
    assert status["status"] == "answers_complete" and status["answered"] == 745
    protocol = status["protocol"]
    items, queries = load_dataset("groupmem")
    assert len(queries) == len({q.qid for q in queries}) == protocol["expected"] == 745
    assert digest([(q.qid, q.question, q.answer) for q in queries]) == protocol["query_hash"]
    assert digest([(i.uid, i.content()) for i in items]) == protocol["content_hash"]
    predictions = {p["qid"]: p for p in
                   map(json.loads, (baseline / "predictions.jsonl").read_text().splitlines())}
    assert len(predictions) == 745 and set(predictions) == {q.qid for q in queries}
    domains = list(dict.fromkeys(q.domain for q in queries))
    cases, observations = [], []
    for domain in domains:
        current = [i for i in items if i.domain == domain]
        texts = {i.uid: i.content() for i in current}
        host = SparseStaticIndex(current)
        for q in [q for q in queries if q.domain == domain]:
            candidates = {str(k): [[u, float(s)] for u, s in host.search(q.question, k)]
                          for k in (10, 100)}
            ids = [uid for uid, _ in candidates["10"]]
            contexts, messages = grounded_prompt(ids, texts, q.question)
            row = predictions[q.qid]
            assert row["question"] == q.question and row["references"] == [str(q.answer)]
            assert row["context_ids"] == ids and row["context_hash"] == digest(contexts)
            assert row["prompt_hash"] == digest(messages), q.qid
            observations.append(dict(qid=q.qid, group=domain, feedback_ids=ids,
                                     messages=traffic_observation(q.question, ids, texts)))
            cases.append(dict(qid=q.qid, group=domain, question=q.question, reference=q.answer,
                              qtype=q.qtype, feedback_ids=ids, candidates=candidates,
                              static=dict(ids=ids, messages=messages, prompt_hash=digest(messages))))
        print("GROUPMEM_NATIVE_REPLAY", domain, sum(q.domain == domain for q in queries), flush=True)
    assert len(cases) == len(observations) == 745
    write_json(PREPARED, dict(status="prepared", domains=domains, cases=cases))
    write_json(OBSERVATIONS, dict(status="audited_observations", cases=observations))
    print("GROUPMEM_PREPARED", len(cases), flush=True)


def cmd_students(args):
    assert args.seed in SEEDS
    source = json.loads(PREPARED.read_text())
    assert source["status"] == "prepared"
    cases = source["cases"]
    training = RESULTS / f"student_s{args.seed}"
    status = json.loads((training / "status.json").read_text())
    assert status["status"] == "complete" and status["seed"] == args.seed
    target = RESULTS / f"groupmem_pairs_s{args.seed}.json"
    if target.exists():
        assert json.loads(target.read_text())["status"] == "prepared"
        print("GROUPMEM_STUDENT_PAIRS_EXIST", args.seed, flush=True)
        return
    import torch
    from src.policy.cumulative_actor import CumulativeMaskActor
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    import os
    model = os.environ.get("AMIE_POLICY_MODEL")
    if not model:
        raise ValueError("Set AMIE_POLICY_MODEL to a local Qwen3-8B checkpoint directory")
    actor = CumulativeMaskActor(model)
    actor.load(training / "adapter")
    items, queries = load_dataset("groupmem")
    assert {q.qid for q in queries} == {c["qid"] for c in cases}
    texts = {i.uid: i.content() for i in items}

    import numpy as np

    class AllSuppress:
        def probabilities(self, messages, count):
            return np.ones(count)

    from src.runs.common import CachedHost
    kwargs = {k: POLICY[k] for k in ("strength", "kappa", "pool")}
    result = []
    for group in source["domains"]:
        selected = [c for c in cases if c["group"] == group]
        host = CachedHost([dict(question=c["question"], candidates=c["candidates"]) for c in selected])
        student, count = NativeSuppression(host, **kwargs), NativeSuppression(host, **kwargs)
        for index, case in enumerate(selected):
            assert [u for u, _ in host.search(case["question"], 10)] == case["feedback_ids"]
            condition, mask = student_step(student, actor, case, texts, POLICY["threshold"])
            count_condition, _ = student_step(count, AllSuppress(), case, texts, POLICY["threshold"])
            if index == 0:
                assert condition["prompt_hash"] == count_condition["prompt_hash"] == case["static"]["prompt_hash"]
            result.append(dict(qid=case["qid"], group=group, question=case["question"],
                               reference=case["reference"], qtype=case["qtype"],
                               conditions=dict(static=case["static"], count=count_condition,
                                               student=condition),
                               student_mask=mask))
        print("GROUPMEM_STUDENT_PREPARED", args.seed, group, len(selected), flush=True)
    assert len(result) == 745
    write_json(target, dict(status="prepared", seed=args.seed,
                            prepared_sha256=sha(PREPARED), cases=result))
    print("GROUPMEM_STUDENT_PAIRS_COMPLETE", args.seed, flush=True)


def cmd_answers(args):
    assert args.seed in SEEDS
    prepared = RESULTS / f"groupmem_pairs_s{args.seed}.json"
    source = json.loads(prepared.read_text())
    assert source["status"] == "prepared"
    _, queries = load_dataset("groupmem")
    gold = {q.qid: q for q in queries}
    cases = source["cases"]
    assert len(cases) == len({c["qid"] for c in cases}) == len(gold) == 745
    prompts = {}
    for c in cases:
        q = gold[c["qid"]]
        assert c["question"] == q.question and c["reference"] == q.answer and c["group"] == q.domain
        assert set(c["conditions"]) == {"static", "count", "student"}
        for condition in c["conditions"].values():
            key = digest(condition["messages"])
            assert key == condition["prompt_hash"]
            if key in prompts:
                assert prompts[key] == condition["messages"]
            prompts[key] = condition["messages"]
    health = answer_health(args.endpoint)
    target = RESULTS / f"groupmem_answers_s{args.seed}.json"
    plan = dict(source_sha256=sha(prepared), generator=health, max_tokens=128, batch_size=1)
    result = dict(status="running", plan=plan, answers={}, rows=[])
    if target.exists():
        result = json.loads(target.read_text())
        assert result["plan"] == plan
    write_json(target, result)
    for key, messages in prompts.items():
        if key in result["answers"]:
            continue
        generated = answer_generate([messages], 128, endpoint=args.endpoint)
        assert isinstance(generated["answers"][0], str)
        result["answers"][key] = generated["answers"][0]
        write_json(target, result)
        if len(result["answers"]) % 25 == 0:
            print("GROUPMEM_GENERATED", args.seed, len(result["answers"]), "/", len(prompts), flush=True)
    result["rows"] = [dict(qid=c["qid"], domain=c["group"], qtype=c["qtype"],
                           question=c["question"], reference=c["reference"],
                           predictions={name: result["answers"][cond["prompt_hash"]]
                                        for name, cond in c["conditions"].items()},
                           prompt_hashes={name: cond["prompt_hash"]
                                          for name, cond in c["conditions"].items()})
                      for c in cases]
    result["status"] = "answers_complete"
    write_json(target, result)
    print("GROUPMEM_ANSWERS_COMPLETE", args.seed, flush=True)


def judge_protocol(ns, hashes):
    root = Path(__file__).resolve().parents[2] / "dataset/groupmem/GroupMemBench"
    prompt = root / "prompts/hipporag_judge_system.txt"
    return dict(model=JUDGE_MODEL, provider="Friday",
                revision="service model alias; response model retained",
                native_sources=hashes, judge_prompt_sha256=sha(prompt),
                completion_budget=2000, temperature=0.2,
                judge="GPT-5, the benchmark's default judge, served via the Friday gateway; native prompt and parser retained",
                metric="native parse_judgment(final) is True; unparseable nonempty judgments count as incorrect",
                scope="745 questions; Qwen3-32B grounded-v1 answers, native judge prompt/parser"), prompt.read_text().strip()


def cmd_judge(args):
    seed = args.seed
    assert seed in SEEDS
    ns, hashes = native_functions()
    protocol, system = judge_protocol(ns, hashes)
    answers_path = RESULTS / f"groupmem_answers_s{seed}.json"
    answers = json.loads(answers_path.read_text())
    assert answers["status"] == "answers_complete"
    pairs_path = RESULTS / f"groupmem_pairs_s{seed}.json"
    cases = {c["qid"]: c for c in json.loads(pairs_path.read_text())["cases"]}
    _, queries = load_dataset("groupmem")
    gold = {q.qid: q for q in queries}
    assert len(answers["rows"]) == 745 and set(cases) == set(gold)
    jobs = []
    for arm in ("static", "count", "student"):
        rows = []
        for row in answers["rows"]:
            q = gold[row["qid"]]
            case = cases[row["qid"]]
            expected = digest(case["conditions"][arm]["messages"])
            assert expected == row["prompt_hashes"][arm] == case["conditions"][arm]["prompt_hash"]
            assert row["predictions"][arm] == answers["answers"][expected]
            rows.append(dict(qid=q.qid, question=q.question, reference=q.answer,
                             prediction=row["predictions"][arm]))
        jobs.append((arm, rows))
    key, base = friday_credentials()
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / f"judgments_s{seed}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row = json.loads(line)
            assert row["protocol"] == protocol
            assert row["key"] == digest(dict(protocol=protocol, body=row["body"]))
            response = row["response"]
            assert response["model"] == JUDGE_MODEL and response["choices"][0]["finish_reason"] == "stop"
            _, final = ns["split_reasoning_and_final"](response["choices"][0]["message"]["content"])
            label = ns["parse_judgment"](final)
            assert row["label"] == label and row["correct"] == (label is True)
            cache[row["key"]] = row

    def request(body):
        identity = digest(dict(protocol=protocol, body=body))
        if identity in cache:
            return identity
        for attempt in range(6):
            req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": "Bearer " + key,
                                                  "Content-Type": "application/json"})
            try:
                with urllib.request.build_opener(NoRedirect()).open(req, timeout=180) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                text = choice["message"].get("content")
                assert text and choice["finish_reason"] == "stop", "Missing or truncated judge response"
                assert result.get("model") == JUDGE_MODEL, "Unexpected judge model"
                _, final = ns["split_reasoning_and_final"](text)
                label = ns["parse_judgment"](final)
                entry = dict(key=identity, protocol=protocol, body=body, response=result,
                             label=label, correct=label is True)
                with cache_path.open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                cache[identity] = entry
                return identity
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError("Judge HTTP " + str(exc.code)) from None
                if attempt == 5:
                    raise RuntimeError("Judge retries exhausted") from None
                time.sleep(60 if getattr(exc, "code", None) == 429 else min(30, 5 * (attempt + 1)))

    results = []
    for arm, rows in jobs:
        judged = []
        for row in rows:
            _, prediction = ns["split_reasoning_and_final"](row["prediction"])
            user = f'Question:\n{row["question"]}\n\nGold Answer:\n{row["reference"]}\n\nAgent Answer:\n{prediction}\n'
            body = ns["build_chat_completion_kwargs"](model=JUDGE_MODEL,
                                                      messages=[dict(role="system", content=system),
                                                                dict(role="user", content=user)],
                                                      max_tokens=2000, temperature=.2,
                                                      reasoning_effort="low")
            identity = request(body)
            entry = cache[identity]
            judged.append(dict(qid=row["qid"], key=identity, correct=entry["correct"],
                               label=entry["label"], domain=gold[row["qid"]].domain,
                               qtype=gold[row["qid"]].qtype))
        score = sum(r["correct"] for r in judged) / len(judged)
        results.append(dict(method=arm, n=len(judged), score=score, rows=judged))
        print("GROUPMEM_JUDGED", seed, arm, len(judged), score, flush=True)
    write_json(RESULTS / f"groupmem_scores_s{seed}.json",
               dict(status="complete", protocol=protocol, seed=seed, results=results))
    print("GROUPMEM_JUDGE_COMPLETE", seed, flush=True)


def cmd_summary(args):
    """Seed-averaged paired effects: average the three fixed seeds within each
    question before comparing (seeds are not independent trials)."""
    per_seed = {}
    for seed in SEEDS:
        path = RESULTS / f"groupmem_scores_s{seed}.json"
        scored = json.loads(path.read_text())
        assert scored["status"] == "complete"
        per_seed[seed] = {r["method"]: r["rows"] for r in scored["results"]}
    control = per_seed[SEEDS[0]]["static"]
    order = [r["qid"] for r in control]
    for seed in SEEDS:
        for method in ("static", "count", "student"):
            rows = per_seed[seed][method]
            assert [r["qid"] for r in rows] == order
            assert all(a["domain"] == b["domain"] for a, b in zip(control, rows))

    def seed_mean_rows(method):
        rows = []
        for i, qid in enumerate(order):
            values = [per_seed[s][method][i]["correct"] for s in SEEDS]
            assert all(type(v) is bool for v in values)
            rows.append(dict(qid=qid, domain=control[i]["domain"], score=statistics.mean(values)))
        return rows

    def paired(on):
        domains = defaultdict(list)
        differences = []
        for i, row in enumerate(control):
            assert type(row["correct"]) is bool
            delta = on[i]["score"] - int(row["correct"])
            domains[row["domain"]].append(delta)
            differences.append(delta)
        assert set(domains) == set(DOMAINS)
        deleted = {name: statistics.mean([d for row, d in zip(control, differences) if row["domain"] != name])
                   for name in domains}
        return dict(questions=len(order), seeds=sorted(SEEDS), shared_domains=4,
                    delta=statistics.mean(differences),
                    per_domain={name: dict(questions=len(values), delta=statistics.mean(values))
                                for name, values in domains.items()},
                    leave_one_domain_out=deleted,
                    domain_deletion_range=[min(deleted.values()), max(deleted.values())],
                    boundary="Average three fixed seeds within each question before paired comparison; seeds are not independent trials. Domain deletion is descriptive sensitivity, not a confidence interval.")

    summary = dict(status="complete", seeds=list(SEEDS),
                   count_minus_off=paired(seed_mean_rows("count")),
                   student_minus_off=paired(seed_mean_rows("student")),
                   per_seed_scores={str(s): {m: sum(r["correct"] for r in per_seed[s][m]) / len(per_seed[s][m])
                                             for m in ("static", "count", "student")}
                                    for s in SEEDS})
    write_json(RESULTS / "groupmem_paired_summary.json", summary)
    print(json.dumps({k: v["delta"] for k, v in summary.items() if k.endswith("minus_off")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--baseline", default=str(RESULTS / "off_answers/groupmem_static_bm25"))
    p.set_defaults(fn=cmd_prepare)
    s = sub.add_parser("students")
    s.add_argument("--seed", type=int, required=True)
    s.set_defaults(fn=cmd_students)
    a = sub.add_parser("answers")
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--endpoint", default=ANSWER_ENDPOINT)
    a.set_defaults(fn=cmd_answers)
    j = sub.add_parser("judge")
    j.add_argument("--seed", type=int, required=True)
    j.set_defaults(fn=cmd_judge)
    m = sub.add_parser("summary")
    m.set_defaults(fn=cmd_summary)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
