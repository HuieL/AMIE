"""The traffic-feedback observation contract shared by all three AMIE policies.

Every policy (count, frozen, trained) consumes the same answer-free observation:
the query's first 300 characters and the first 200 characters of each memory in
the host's native top ten. The frozen policy's system prompt and the verdict
parser are fixed here so the teacher masks, the frozen arm, and the distilled
student are provably the same interface.
"""
from __future__ import annotations

import json

TRAFFIC_SYSTEM = (
    "You maintain the access index of an agent's long-term memory. A query was "
    "issued and the retriever returned the memories below. Decide, for each, "
    "whether it was retrieved DESERVEDLY (it bears on the query) or whether it "
    "INTRUDED (it was surfaced by lexical accident, generic phrasing or topical "
    "adjacency and does not bear on the query).\n"
    "Your verdict does not reorder these results. It is written into a "
    "persistent penalty that affects which memories this agent can reach for "
    "ALL FUTURE queries, so judge the memory's tendency to intrude rather than "
    "its rank here, and prefer 'noop' when unsure: suppressing a memory that "
    "merely ranked low costs future access to it.\n"
    'Reply with JSON only: {"verdicts": [{"id": <int>, "a": "suppress"|"noop"}, '
    "...]} listing only the memories you judge to have intruded, plus nothing "
    "else. An empty list is a valid and often correct answer."
)

CORRECTION_SUFFIX = (
    "Return only JSON verdicts with unique integer IDs and suppress/noop actions."
)

MAX_FORMAT_ATTEMPTS = 3


def observation(query, feedback_ids, texts):
    """Exact policy input: system prompt plus clipped query/retrieved listing."""
    return [
        dict(role="system", content=TRAFFIC_SYSTEM),
        dict(role="user", content="QUERY: " + query[:300] + "\n\nRETRIEVED:\n" +
             "\n".join(f"{i}. " + texts[u][:200] for i, u in enumerate(feedback_ids))),
    ]


def parse_mask(text, n):
    """Parse a verdict response into a binary mask over the n feedback slots."""
    verdicts = json.loads(text)["verdicts"]
    if not isinstance(verdicts, list):
        raise ValueError("Expected verdict list")
    mask, seen = [0] * n, set()
    for row in verdicts:
        i, action = row["id"], row["a"]
        if type(i) is not int or not 0 <= i < n or i in seen or action not in ("noop", "suppress"):
            raise ValueError("Invalid or repeated verdict")
        seen.add(i)
        mask[i] = int(action == "suppress")
    return mask
