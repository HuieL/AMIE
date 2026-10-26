"""GroupMemBench loader -- the multi-agent memory setting.

Every other benchmark here is single-agent: one user, one assistant, one stream.
GroupMemBench is a set of shared channels, each owned by a different agent, with
many human authors posting into them, threaded by `reply_to`. That makes it the
benchmark for the question a multi-agent system actually faces -- when memory is
written by several agents and read by another, does the reader still reach what
someone else stored -- and it is the setting where the index has a routing job
(which agent's channel holds this) that does not exist single-agent.

Four domains (Technology, Finance, Healthcare, Manufacturing), seven agent
channels each, and six question types including `abstention`, where the correct
behaviour is to decline. Answers are free-text, so accuracy needs the LLM judge
(`src/evaluation/qa.py`); there is no supporting-message annotation, so the
judge-free retrieval protocol does not apply here and we do not fake one.

`is_noise` marks messages the benchmark inserted as distractors. They are kept
in the store -- removing them would be tuning the store to the benchmark -- but
are exposed so the interference analysis can condition on them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "dataset/groupmem/GroupMemBench"
DOMAINS = ("Technology", "Finance", "Healthcare", "Manufacturing")
QTYPES = ("multi_hop", "temporal", "knowledge_update", "term_ambiguity",
          "user_implicit", "abstention")


@dataclass
class GroupItem:
    uid: str
    text: str
    scope: str                  # domain:agent -- the channel
    domain: str
    agent: str
    author: str = ""
    role: str = ""
    reply_to: str = ""
    timestamp: str = ""
    stage: str = ""
    is_noise: bool = False
    turn_index: int = 0
    sources: list = field(default_factory=list)

    def content(self) -> str:
        who = f"{self.author} ({self.role})" if self.role else self.author
        return f"[{self.agent}] {who}: {self.text}"


@dataclass
class GroupQuery:
    qid: str
    question: str
    answer: str
    domain: str
    qtype: str
    asking_user: str = ""
    gold_uids: list = field(default_factory=list)

    @property
    def conv_id(self):
        return self.domain

    @property
    def is_abstention(self):
        return self.qtype == "abstention"


def load_groupmem(domains=DOMAINS, qtypes=QTYPES, root=ROOT):
    root = Path(root)
    items, queries = [], []
    order = 0
    for dom in domains:
        f = root / "data/final" / dom / f"synthetic_domain_channels_rolevariants_{dom}.json"
        if not f.exists():
            continue
        raw = json.loads(f.read_text())
        for agent, msgs in raw.items():
            for m in msgs:
                node = m.get("msg_node")
                if not node:
                    continue
                uid = f"{dom}:{agent}::{node}"
                items.append(GroupItem(
                    uid=uid, text=str(m.get("content", "")),
                    scope=f"{dom}:{agent}", domain=dom, agent=agent,
                    author=str(m.get("author", "")), role=str(m.get("role", "")),
                    reply_to=str(m.get("reply_to") or ""),
                    timestamp=str(m.get("timestamp", "")),
                    stage=str(m.get("stage", "")),
                    is_noise=bool(m.get("is_noise", False)),
                    turn_index=order, sources=[uid]))
                order += 1
        for qt in qtypes:
            qf = root / "questions" / dom / f"{qt}.jsonl"
            if not qf.exists():
                continue
            for line in qf.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                queries.append(GroupQuery(
                    qid=f"{dom}::{d.get('id')}", question=str(d.get("question", "")),
                    answer=str(d.get("answer", "")), domain=dom, qtype=qt,
                    asking_user=str(d.get("asking_user_id", ""))))
    return items, queries


def reply_edges(items):
    """(source_idx, target_idx) pairs from the channel's own thread structure.

    A multi-agent channel carries an explicit discourse graph that a single
    dialogue does not, and it is index-side structure: it is known at write time
    and names no content.
    """
    pos = {it.uid: i for i, it in enumerate(items)}
    node_to_uid = {}
    for it in items:
        node_to_uid.setdefault((it.domain, it.agent, it.uid.split("::")[-1]), it.uid)
    out = []
    for i, it in enumerate(items):
        if not it.reply_to:
            continue
        parent = node_to_uid.get((it.domain, it.agent, it.reply_to))
        if parent and parent in pos:
            out.append((pos[parent], i))
    return out


if __name__ == "__main__":
    import collections
    items, queries = load_groupmem()
    print(f"messages : {len(items)}  ({sum(i.is_noise for i in items)} noise)")
    print(f"questions: {len(queries)}")
    print("by domain:", collections.Counter(q.domain for q in queries))
    print("by type  :", collections.Counter(q.qtype for q in queries))
    print("channels :", len({i.scope for i in items}))
    print("authors  :", len({i.author for i in items}))
    print("reply edges:", len(reply_edges(items)))
