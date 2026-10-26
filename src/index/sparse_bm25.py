"""Sparse BM25 that scales to hundreds of thousands of memories.

`rank_bm25` scores every document in pure Python, which is O(N) per query with a
large constant -- unusable at the store sizes needed to test whether
accessibility actually binds at scale. This implements the same Okapi BM25 with
a CSR term-document matrix so a query costs one sparse mat-vec over only the
posting lists of its terms.

Scores are identical in ranking to BM25Okapi up to floating point.
"""
from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOKEN.findall(text.lower())


class SparseBM25:
    def __init__(self, docs, k1=1.5, b=0.75, min_df=1):
        self.k1, self.b = k1, b
        vocab, rows, cols, vals = {}, [], [], []
        dl = np.zeros(len(docs), dtype=np.float32)
        for di, d in enumerate(docs):
            toks = tokenize(d)
            dl[di] = len(toks)
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                j = vocab.setdefault(t, len(vocab))
                rows.append(di)
                cols.append(j)
                vals.append(c)
        self.vocab = vocab
        self.N = len(docs)
        self.dl = dl
        self.avgdl = float(dl.mean()) if self.N else 1.0

        tf = sp.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)),
                           shape=(self.N, len(vocab)))
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (self.N - df + 0.5) / (df + 0.5)).astype(np.float32)

        # precompute the BM25 term weight per (doc, term): idf is applied at query
        # time so the stored matrix is query-independent
        norm = (self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)).astype(np.float32)
        tf = tf.tocoo()
        w = (tf.data * (self.k1 + 1.0)) / (tf.data + norm[tf.row])
        self.W = sp.csc_matrix((w, (tf.row, tf.col)), shape=(self.N, len(vocab)))

    def scores(self, query):
        out = np.zeros(self.N, dtype=np.float32)
        seen = {}
        for t in tokenize(query):
            j = self.vocab.get(t)
            if j is None:
                continue
            seen[j] = seen.get(j, 0) + 1
        for j in seen:
            col = self.W.getcol(j)
            out[col.indices] += col.data * self.idf[j]
        return out

    def scores_weighted(self, term_weights):
        """BM25 over a weighted bag of terms rather than a literal query string.

        Needed by term-level index evolution: the index adds terms the query did
        not contain, each with its own strength, so the scorer has to accept a
        weighted bag instead of a string.
        """
        out = np.zeros(self.N, dtype=np.float32)
        for t, wt in term_weights.items():
            j = self.vocab.get(t)
            if j is None or wt == 0.0:
                continue
            col = self.W.getcol(j)
            out[col.indices] += col.data * self.idf[j] * float(wt)
        return out

    def term_idf(self, t):
        j = self.vocab.get(t)
        return float(self.idf[j]) if j is not None else 0.0
