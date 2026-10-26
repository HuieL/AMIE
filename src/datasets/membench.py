"""MemBench loader (arXiv 2506.21605).

The most useful benchmark added in this revision, because it annotates both
halves of the measurement this project needs: `target_step_id` gives the
supporting messages, so accessibility is measurable judge-free exactly as on
LoCoMo, and `ground_truth` gives a four-way multiple choice, so task accuracy is
measurable without an LLM judge either. It is the only dataset here where the
retrieval metric and the answer metric can be compared with no judge noise
between them.

Structure. Each file is one *question type* under one *perspective* --
`FirstAgent` is participation (the agent is in the dialogue), `ThirdAgent` is
observation (the agent watches it) -- and holds five categories (`roles`,
`events`, `items`, `places`, `hybrid`) of 500 traces each. A trace is a short
message flow plus one question over it.

Two store modes, matching the two regimes the project separates:

  per_trace : each question sees only its own trace, the published protocol.
  global    : every trace of the selected types is pooled into one store, so a
              question's evidence competes with ~10^5 unrelated memories. This
              is the interference regime, and it is the reason a benchmark of
              short traces is still informative here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "dataset/membench/Membench/MemData"

PERSPECTIVES = ("FirstAgent", "ThirdAgent")
# the eight types present under both perspectives, so the two are comparable
TYPES = ("simple", "comparative", "aggregative", "conditional", "noisy",
         "post_processing", "knowledge_update", "highlevel")
# every category the files use; `None` selects whatever a file happens to hold
CATEGORIES = None

# Two file shapes. `FirstAgent` (and the `highlevel` files of both
# perspectives) store `message_list` as a list of *sessions*, each a list of
# turns carrying `user_message`/`assistant_message`; the rest store one flat
# list of `message` records. Turn ids (`sid`/`mid`) are global across sessions
# in both, and `target_step_id` is either that id or `[id, session_index]`,
# which is what makes a single gold resolution possible.


@dataclass
class MemBenchItem:
    uid: str
    text: str
    scope: str                 # persp:type:category:tid -- the trace
    turn_index: int = 0
    timestamp: str = ""
    place: str = ""
    sources: list = field(default_factory=list)

    def content(self) -> str:
        head = " ".join(x for x in (self.timestamp, self.place) if x)
        return f"[{head}] {self.text}" if head else self.text


@dataclass
class MemBenchQuery:
    qid: str
    question: str
    answer: str
    scope: str
    gold_uids: list = field(default_factory=list)
    choices: dict = field(default_factory=dict)
    ground_truth: str = ""
    qtype: str = ""
    category: str = ""
    perspective: str = ""

    @property
    def conv_id(self):
        return self.scope

    def prompt_choices(self) -> str:
        return "\n".join(f"{k}. {v}" for k, v in sorted(self.choices.items()))


def _turns(message_list):
    """Flatten either file shape into (turn_id, text, time, place)."""
    if not message_list:
        return
    flat = message_list
    if isinstance(message_list[0], list):
        flat = [t for sess in message_list for t in sess]
    for t in flat:
        if not isinstance(t, dict):
            continue
        tid = t.get("mid", t.get("sid"))
        if tid is None:
            continue
        if "message" in t:
            text = str(t.get("message", ""))
        else:
            u = t.get("user_message", t.get("user", "")) or ""
            a = t.get("assistant_message", t.get("assistant", "")) or ""
            text = f"{u} {a}".strip()
        yield int(tid), text, str(t.get("time", "")), str(t.get("place", ""))


def _gold_ids(target_step_id):
    """`[[119, 5], ...]` and `[10, ...]` both resolve to the global turn id."""
    for g in (target_step_id or []):
        if isinstance(g, (list, tuple)):
            if g:
                yield int(g[0])
        else:
            yield int(g)


def load_membench(perspectives=PERSPECTIVES, types=TYPES, categories=CATEGORIES,
                  max_traces=None, root=ROOT):
    """Returns (items, queries) pooled over every requested slice."""
    items, queries = [], []
    order = 0
    for persp in perspectives:
        for qtype in types:
            path = Path(root) / persp / f"{qtype}.json"
            if not path.exists():
                continue
            raw = json.loads(path.read_text())
            cats = categories if categories is not None else sorted(raw.keys())
            for cat in cats:
                traces = raw.get(cat)
                if not traces:
                    continue
                if max_traces:
                    traces = traces[:max_traces]
                for tr in traces:
                    if not isinstance(tr, dict):
                        continue
                    scope = f"{persp}:{qtype}:{cat}:{tr.get('tid')}"
                    by_mid = {}
                    for tid, text, ts, place in _turns(tr.get("message_list")):
                        uid = f"{scope}::m{tid}"
                        by_mid[tid] = uid
                        items.append(MemBenchItem(
                            uid=uid, text=text, scope=scope, turn_index=order,
                            timestamp=ts, place=place, sources=[uid]))
                        order += 1
                    qa = tr.get("QA") or {}
                    if not qa.get("question"):
                        continue
                    gold = [by_mid[m] for m in _gold_ids(qa.get("target_step_id"))
                            if m in by_mid]
                    queries.append(MemBenchQuery(
                        qid=f"{scope}::q{qa.get('qid', 0)}",
                        question=str(qa.get("question", "")),
                        answer=str(qa.get("answer", "")),
                        scope=scope, gold_uids=gold,
                        choices=dict(qa.get("choices") or {}),
                        ground_truth=str(qa.get("ground_truth", "")),
                        qtype=qtype, category=cat, perspective=persp))
    return items, queries


def answerable(queries, items):
    have = {i.uid for i in items}
    return [q for q in queries if q.gold_uids and any(g in have for g in q.gold_uids)]


def per_trace_stores(items, queries):
    """Iterator of (scope, trace_items, trace_queries) for the published mode."""
    by_scope = {}
    for it in items:
        by_scope.setdefault(it.scope, []).append(it)
    q_by_scope = {}
    for q in queries:
        q_by_scope.setdefault(q.scope, []).append(q)
    for scope in sorted(by_scope):
        yield scope, by_scope[scope], q_by_scope.get(scope, [])


if __name__ == "__main__":
    import collections
    items, queries = load_membench()
    ans = answerable(queries, items)
    print(f"memories : {len(items)}")
    print(f"questions: {len(queries)}  (with gold: {len(ans)})")
    print("by perspective:", collections.Counter(q.perspective for q in ans))
    print("by type      :", collections.Counter(q.qtype for q in ans))
    print("by category  :", collections.Counter(q.category for q in ans))
    g = [len(q.gold_uids) for q in ans]
    print(f"gold/query   : mean {sum(g)/max(len(g),1):.2f}, max {max(g) if g else 0}")
    mcq = [q for q in ans if q.choices and q.ground_truth]
    print(f"with MCQ     : {len(mcq)}")
    print("traces       :", len({i.scope for i in items}))
