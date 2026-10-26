"""Lexical host index: Okapi BM25 over a fixed content store.

The central abstraction of the paper: memory is M = (C, I). The host here keeps
C fixed (same items, same text, nothing rewritten or summarised); AMIE wraps the
host's access structure I without touching content, which is what makes the
comparison controlled.

`StaticIndex` is the reference rank_bm25 backend; `SparseStaticIndex` is the
same Okapi BM25 through a CSR backend (rankings identical up to floating point)
and is the host used by every reported result.
"""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return _TOKEN.findall(text.lower())


class StaticIndex:
    """BM25 over the content store. The access structure never changes."""

    name = "static"

    def __init__(self, items):
        self.items = list(items)
        self.uids = [i.uid for i in self.items]
        self.pos = {u: k for k, u in enumerate(self.uids)}
        self._fit([i.content() for i in self.items])

    def _fit(self, texts):
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def base_scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)

    def search(self, query: str, k: int = 10):
        s = self.base_scores(query)
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        return [(self.uids[j], float(s[j])) for j in top]

    def observe(self, *args, **kwargs):
        return None


class SparseBackendMixin:
    """Same Okapi BM25, CSR-backed, for stores past ~10^5 memories."""

    def _fit(self, texts):
        from .sparse_bm25 import SparseBM25
        self._bm25 = SparseBM25(list(texts))

    def base_scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.scores(query), dtype=float)


class SparseStaticIndex(SparseBackendMixin, StaticIndex):
    name = "static"
