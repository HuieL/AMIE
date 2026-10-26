"""LoCoMo loader.

LoCoMo gives us something most memory benchmarks do not: per-question *gold
evidence* at turn granularity (`"evidence": ["D1:3", ...]` referencing each
turn's `dia_id`). That lets us measure memory **accessibility** directly --
whether the supporting turn is still reachable as the store grows -- without an
LLM judge in the loop.

Note `dia_id` is only unique *within* a conversation, so memory item ids are
(conversation_id, dia_id).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "dataset/locomo/locomo/data/locomo10.json"

# LoCoMo category 5 is the adversarial / unanswerable split: there is no
# supporting evidence to retrieve, so it is excluded from accessibility metrics.
ADVERSARIAL_CATEGORY = 5


@dataclass
class MemoryItem:
    """One atomic unit of stored experience (here: a dialogue turn)."""
    uid: str                  # globally unique: "<conv_id>::<dia_id>"
    conv_id: str
    dia_id: str
    speaker: str
    text: str
    session: int
    timestamp: str = ""
    turn_index: int = 0       # global arrival order == "creation time"

    def content(self) -> str:
        return f"{self.speaker}: {self.text}"


@dataclass
class Query:
    qid: str
    conv_id: str
    question: str
    answer: str
    category: int
    gold_uids: list = field(default_factory=list)

    @property
    def is_adversarial(self) -> bool:
        return self.category == ADVERSARIAL_CATEGORY


def render_locomo_item(item, *, include_session_dates=False):
    """Versioned host input; preserve benchmark temporal context when requested."""
    if not include_session_dates:
        return item.content()
    if not item.timestamp.strip():
        raise ValueError(f'Missing session timestamp for {item.uid}')
    return f'[Session {item.session}; date/time: {item.timestamp}]\n{item.content()}'


def _session_ids(conv: dict):
    out = []
    for k in conv:
        if k.startswith("session_") and not k.endswith("date_time"):
            try:
                out.append((int(k.split("_")[1]), k))
            except (ValueError, IndexError):
                continue
    return sorted(out)


def load_locomo(path=DEFAULT_PATH):
    """Returns (items, queries). Items are in conversation/session/turn order."""
    raw = json.loads(Path(path).read_text())
    items, queries = [], []
    order = 0
    for ci, sample in enumerate(raw):
        conv_id = str(sample.get("sample_id", ci))
        conv = sample["conversation"]
        for sess_num, key in _session_ids(conv):
            turns = conv.get(key)
            if not isinstance(turns, list):
                continue
            ts = conv.get(f"{key}_date_time", "")
            for turn in turns:
                dia = turn.get("dia_id")
                if not dia:
                    continue
                items.append(MemoryItem(
                    uid=f"{conv_id}::{dia}", conv_id=conv_id, dia_id=dia,
                    speaker=turn.get("speaker", ""), text=turn.get("text", ""),
                    session=sess_num, timestamp=ts, turn_index=order))
                order += 1
        for qi, qa in enumerate(sample.get("qa", [])):
            gold = [f"{conv_id}::{e}" for e in (qa.get("evidence") or [])
                    if isinstance(e, str)]
            queries.append(Query(
                qid=f"{conv_id}::q{qi}", conv_id=conv_id,
                question=qa.get("question", ""),
                answer=str(qa.get("answer", qa.get("adversarial_answer", ""))),
                category=int(qa.get("category", -1)), gold_uids=gold))
    return items, queries


def answerable(queries, items):
    """Queries with >=1 gold turn that actually exists in the store."""
    have = {i.uid for i in items}
    return [q for q in queries
            if not q.is_adversarial and q.gold_uids
            and any(g in have for g in q.gold_uids)]


if __name__ == "__main__":
    items, queries = load_locomo()
    ans = answerable(queries, items)
    convs = sorted({i.conv_id for i in items})
    print(f"conversations : {len(convs)}")
    print(f"memory items  : {len(items)}")
    print(f"queries       : {len(queries)}  (answerable w/ gold: {len(ans)})")
    import collections
    print("per-conv items:", collections.Counter(i.conv_id for i in items).most_common(3), "...")
    g = [len(q.gold_uids) for q in ans]
    print(f"gold turns/query: mean {sum(g)/len(g):.2f}, max {max(g)}")
