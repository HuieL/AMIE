"""Cumulative suppression preserving the host's exact disabled top-k identity."""
from collections import defaultdict
import math


class NativeSuppression:
    def __init__(self, host, *, strength=0.5, kappa=3, pool=100):
        if not 0 <= strength < 1 or kappa <= 0 or pool < 10:
            raise ValueError('Invalid fixed suppression configuration')
        self.host, self.strength, self.kappa, self.pool = host, strength, kappa, pool
        self.counts = defaultdict(int)

    def search(self, query, k=10):
        native = list(self.host.search(query, k))
        if self.strength == 0 or not any(self.counts[u] for u, _ in native):
            return native
        # Some native argpartition implementations change tied IDs with k.
        # Anchor the native top-k before extending the candidate pool so an
        # unchanged score retains native precedence, including at a zero tie.
        candidates, seen = [], set()
        for uid, score in native + list(self.host.search(query, self.pool)):
            if not math.isfinite(score) or score < 0:
                raise ValueError('Native suppression requires finite nonnegative scores')
            if uid not in seen:
                candidates.append((uid, score))
                seen.add(uid)
            if len(candidates) >= self.pool:
                break
        rescored = [(u, s * (1 - self.strength * self.counts[u] / (self.counts[u] + self.kappa)))
                    for u, s in candidates]
        return sorted(rescored, key=lambda row: -row[1])[:k]

    def observe(self, ids, mask):
        if len(ids) != len(mask) or len(set(ids)) != len(ids) or any(v not in (0, 1) for v in mask):
            raise ValueError('Invalid semantic mask')
        for uid, selected in zip(ids, mask):
            if selected:
                self.counts[uid] += 1
