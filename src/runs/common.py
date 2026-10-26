"""Shared protocol constants and helpers for every AMIE experiment stage.

One module holds the contracts all pipelines share: the fixed answer
generator identity, the grounded prompt shapes, the policy observation
interface, the replay helpers, and the two bootstrap conventions (F1
contrasts: numpy seed 20260914, conversation-cluster primary plus
question-iid secondary; recall replays: random.Random 20260920, percent
units). Nothing here calls a model.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import urllib.request
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

ANSWER_ENDPOINT = os.environ.get("AMIE_ANSWER_ENDPOINT", "http://127.0.0.1:8034")
ANSWER_MODEL = os.environ.get("AMIE_ANSWER_MODEL", "Qwen/Qwen3-32B")
ANSWER_REVISION = os.environ.get("AMIE_ANSWER_REVISION",
                                 "9216db5781bf21249d130ec9da846c4624c16137")
TEACHER_MODEL = os.environ.get("AMIE_TEACHER_MODEL", "GPT-5.6-Sol")
FROZEN_MODEL = os.environ.get("AMIE_FROZEN_MODEL", "Qwen3-8B")

# Fixed policy constants (paper Section 3): score * (1 - strength * u/(u+kappa)).
POLICY = dict(strength=0.5, kappa=3, prior_weight=0, pool=100, feedback_k=10, threshold=0.5)

GROUNDED_SYSTEM = ("Answer the task using only the provided memories. "
                   "If unsupported, say INSUFFICIENT. Answer briefly without explanation.")
MCQ_SYSTEM = ("Answer from the provided memories. Return a JSON object with exactly one "
              'field, "choice", whose value is A, B, C, or D.')

SEEDS = (20260908, 20260909, 20260910)
FIT_GROUPS = ("conv-30", "conv-41", "conv-42", "conv-44", "conv-48", "conv-49")

F1_BOOT_DRAWS = 10000
F1_BOOT_SEED = 20260914
RECALL_BOOT_DRAWS = 10000
RECALL_BOOT_SEED = 20260920


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def load_dataset(name):
    """(items, queries); LoCoMo keeps answerable categories 1-4 only."""
    if name == "locomo":
        from src.datasets.locomo import load_locomo
        items, queries = load_locomo()
        queries = [q for q in queries if not q.is_adversarial]
    elif name == "groupmem":
        from src.datasets.groupmem import load_groupmem
        items, queries = load_groupmem()
    elif name == "membench":
        from src.datasets.membench import load_membench
        items, queries = load_membench()
    else:
        raise ValueError(f"Unknown dataset {name}")
    if not items or not queries:
        raise ValueError("Empty benchmark payload")
    if len({q.qid for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs")
    return items, queries


def answer_health(endpoint=ANSWER_ENDPOINT):
    with urllib.request.urlopen(endpoint + "/health", timeout=30) as response:
        health = json.load(response)
    if health.get("model") != ANSWER_MODEL or health.get("revision") != ANSWER_REVISION:
        raise ValueError(f"Wrong answer model: {health}")
    return health


def answer_generate(conversations, max_tokens, endpoint=ANSWER_ENDPOINT, decoding=None):
    """One batch against the pinned local answer service."""
    health = answer_health(endpoint)
    payload = dict(model=ANSWER_MODEL, conversations=conversations, max_tokens=max_tokens)
    if decoding:
        payload["decoding"] = decoding
    request = urllib.request.Request(endpoint + "/generate", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    expected = dict(health)
    if decoding:
        expected["decoding"] = decoding
    if len(result["answers"]) != len(conversations) or result["protocol"] != expected:
        raise ValueError("Generator response mismatch")
    return result


def grounded_prompt(ids, texts, question, mcq_choices=None):
    """Sealed answer context: native top-ten, 120-word truncation per memory."""
    contexts = [" ".join(texts[u].split()[:120]) for u in ids]
    context = "\n".join(f"[{i + 1}] {text}" for i, text in enumerate(contexts)) or "(no memories retrieved)"
    if mcq_choices:
        system = MCQ_SYSTEM
        question = question + "\n" + "\n".join(f"{k}. {v}" for k, v in sorted(mcq_choices.items()))
    else:
        system = GROUNDED_SYSTEM
    messages = [dict(role="system", content=system),
                dict(role="user", content=f"Memories:\n{context}\n\nQuestion:\n{question}")]
    return contexts, messages


class CachedHost:
    """Host backed by saved candidates; identical scores, zero recomputation."""

    def __init__(self, rows):
        self.rows = {}
        for row in rows:
            value = row["candidates"]
            if row["question"] in self.rows:
                assert self.rows[row["question"]] == value
            self.rows[row["question"]] = value

    def search(self, query, k):
        return [(u, float(s)) for u, s in self.rows[query][str(k)]]


def replay_references(host, cases):
    """Re-verify sealed frozen/count condition IDs against a fresh replay."""
    from src.policy.native_suppression import NativeSuppression
    frozen, count = NativeSuppression(host), NativeSuppression(host)
    for c in cases:
        ids = [u for u, _ in host.search(c["question"], 10)]
        assert ids == c["feedback_ids"] == c["conditions"]["static"]["ids"]
        assert [u for u, _ in frozen.search(c["question"])] == c["conditions"]["frozen"]["ids"]
        assert [u for u, _ in count.search(c["question"])] == c["conditions"]["count"]["ids"]
        frozen.observe(ids, c["mask"])
        count.observe(ids, [1] * len(ids))


def student_step(student, actor, case, texts, threshold):
    """One released-order step: seal the current answer context, then update.

    The current answer sees the pre-update state, irrespective of this mask.
    """
    from src.policy.traffic import observation
    ids = [u for u, _ in student.search(case["question"])]
    _, messages = grounded_prompt(ids, texts, case["question"])
    condition = dict(ids=ids, messages=messages, prompt_hash=digest(messages))
    feedback = case["feedback_ids"]
    obs = observation(case["question"], feedback, texts)
    mask = (actor.probabilities(obs, len(feedback)) >= threshold).astype(int).tolist()
    student.observe(feedback, mask)
    return condition, mask


def locomo_world():
    """LoCoMo items/queries plus evidence gold and released per-conversation order."""
    items, queries = load_dataset("locomo")
    gold, cat = {}, {}
    for sample in json.loads((ROOT / "dataset/locomo/locomo/data/locomo10.json").read_text()):
        g = sample["sample_id"]
        avail = {f'{g}::{t["dia_id"]}' for k, ts in sample["conversation"].items()
                 if k.startswith("session_") and isinstance(ts, list) for t in ts if "dia_id" in t}
        for ordinal, q in enumerate(sample["qa"]):
            qid = f"{g}::q{ordinal}"
            cat[qid] = q["category"]
            ids = {f"{g}::{e}" for e in (q.get("evidence") or [])}
            if ids and ids <= avail:
                gold[qid] = ids
    groups, by_conv = defaultdict(list), defaultdict(list)
    for item in items:
        groups[item.conv_id].append(item)
    for q in queries:
        by_conv[q.conv_id].append(q)
    for g in by_conv:
        by_conv[g].sort(key=lambda q: int(q.qid.split("::q")[1]))
    return dict(items=items, queries=queries, gold=gold, cat=cat, groups=groups, by_conv=by_conv)


def locomo_gold_qa():
    """qid -> raw QA record for every LoCoMo question (all categories)."""
    raw = json.loads((ROOT / "dataset/locomo/locomo/data/locomo10.json").read_text())
    return {f"{s['sample_id']}::q{i}": q for s in raw for i, q in enumerate(s["qa"])}


def recall(ids, gold_ids):
    return len(set(ids) & set(gold_ids)) / len(set(gold_ids))


def cluster_boot(per_q, draws=RECALL_BOOT_DRAWS, seed=RECALL_BOOT_SEED):
    """Conversation-cluster bootstrap of the mean of a per-question quantity (percent)."""
    by = defaultdict(list)
    for qid, v in per_q.items():
        by[qid.split("::")[0]].append(v)
    convs = sorted(by)
    rng = random.Random(seed)
    mean = 100 * statistics.mean(v for vs in by.values() for v in vs)
    stats = []
    for _ in range(draws):
        pick = [convs[rng.randrange(len(convs))] for _ in convs]
        stats.append(100 * statistics.mean([v for c in pick for v in by[c]]))
    stats.sort()
    return dict(mean=round(mean, 3), lo=round(stats[int(0.025 * draws)], 3),
                hi=round(stats[int(0.975 * draws) - 1], 3), n=len(per_q))


def f1_contrast_bootstrap(rows, left, right, draws=F1_BOOT_DRAWS, seed=F1_BOOT_SEED):
    """Paired F1 contrast: conversation-cluster primary, question-iid secondary."""
    import numpy as np
    groups = sorted({r["group"] for r in rows})
    sizes = np.array([sum(r["group"] == g for r in rows) for g in groups])
    cluster_draws = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    diffs = np.array([r["scores"][left] - r["scores"][right] for r in rows])
    totals = np.array([sum(d for d, r in zip(diffs, rows) if r["group"] == g) for g in groups])
    clustered = totals[cluster_draws].sum(axis=1) / sizes[cluster_draws].sum(axis=1)
    rng = np.random.default_rng(seed)
    question_draws = []
    for start in range(0, draws, 250):
        indices = rng.integers(0, len(rows), size=(min(250, draws - start), len(rows)))
        question_draws.extend(diffs[indices].mean(axis=1))
    return dict(delta=float(diffs.mean()),
                primary_conversation_cluster_ci95=np.quantile(clustered, [.025, .975]).tolist(),
                secondary_question_iid_ci95=np.quantile(question_draws, [.025, .975]).tolist(),
                conversation_wins=int((totals > 1e-12).sum()),
                conversation_losses=int((totals < -1e-12).sum()),
                conversation_ties=int((abs(totals) <= 1e-12).sum()))
