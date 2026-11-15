from __future__ import annotations

import math
import re
from collections import Counter


class BM25Indexer:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: dict[str, int] = {}
        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: dict[str, Counter[str]] = {}

    def _tokenize(self, text: str) -> list[str]:
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in clean.split() if len(w) > 1]

    def add_document(self, doc_id: str, text: str) -> None:
        tokens = self._tokenize(text)
        length = len(tokens)
        counts = Counter(tokens)

        if doc_id in self.term_freqs:
            old_counts = self.term_freqs[doc_id]
            for word in old_counts:
                self.doc_freqs[word] = max(0, self.doc_freqs.get(word, 1) - 1)
        else:
            self.corpus_size += 1

        self.term_freqs[doc_id] = counts
        self.doc_lengths[doc_id] = length

        for word in counts:
            self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        total_len = sum(self.doc_lengths.values())
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and all of its lexical corpus statistics.

        Safe to call for an absent ID; repeated deletion must not alter IDF.
        """
        counts = self.term_freqs.pop(doc_id, None)
        if counts is None:
            return False

        self.doc_lengths.pop(doc_id)
        for word in counts:
            remaining = self.doc_freqs[word] - 1
            if remaining:
                self.doc_freqs[word] = remaining
            else:
                del self.doc_freqs[word]

        self.corpus_size -= 1
        self.avg_doc_len = (
            sum(self.doc_lengths.values()) / self.corpus_size
            if self.corpus_size else 0.0
        )
        return True

    def score(self, query: str) -> list[tuple[str, float]]:
        q_tokens = self._tokenize(query)
        scores: dict[str, float] = {}

        for q in q_tokens:
            df = self.doc_freqs.get(q, 0)
            if df == 0:
                continue
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, freqs in self.term_freqs.items():
                tf = freqs.get(q, 0)
                if tf == 0:
                    continue
                d_len = self.doc_lengths[doc_id]
                denom = tf + self.k1 * (
                    1.0
                    - self.b
                    + self.b * (d_len / self.avg_doc_len if self.avg_doc_len > 0 else 1.0)
                )
                numerator = tf * (self.k1 + 1.0)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (numerator / denom)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
