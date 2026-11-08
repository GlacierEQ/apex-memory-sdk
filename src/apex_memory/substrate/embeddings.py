from __future__ import annotations

import hashlib
import math
import re


class SubwordEmbeddingEngine:
    def __init__(self, dimensions: int = 128, ngram_range: tuple[int, int] = (2, 4)) -> None:
        self.dimensions = dimensions
        self.min_n, self.max_n = ngram_range

    def _extract_ngrams(self, text: str) -> list[str]:
        clean = re.sub(r"[^\w\s]", "", text.lower())
        tokens = clean.split()
        ngrams: list[str] = []
        for token in tokens:
            t = f"<{token}>"
            for n in range(self.min_n, min(self.max_n + 1, len(t) + 1)):
                for i in range(len(t) - n + 1):
                    ngrams.append(t[i : i + n])
        return ngrams

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        ngrams = self._extract_ngrams(text)
        if not ngrams:
            return vec

        for ng in ngrams:
            digest = hashlib.md5(ng.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            return [x / norm for x in vec]
        return vec

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        if len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 > 0.0 and n2 > 0.0:
            return dot / (n1 * n2)
        return 0.0
