from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class ScoringResult:
    text: str
    risk_probability: float
    top_contributing_features: List[Tuple[str, float]]
    latency_ms: float


class HybridTfidfToxicityClassifier:
    """Word (1-2) + Character (3-5) sublinear TF-IDF vectorizer with L2-regularized Logistic Regression."""

    def __init__(
        self,
        char_ngram_range: Tuple[int, int] = (3, 5),
        learning_rate: float = 0.6,
        l2_reg: float = 1e-3,
        epochs: int = 80,
    ) -> None:
        self.char_min, self.char_max = char_ngram_range
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.epochs = epochs
        self._idf: Dict[str, float] = {}
        self._weights: Dict[str, float] = {}
        self._bias: float = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0.0:
            ez = math.exp(-min(z, 60.0))
            return 1.0 / (1.0 + ez)
        ez = math.exp(max(z, -60.0))
        return ez / (1.0 + ez)

    def _extract_raw_tokens(self, text: str) -> Counter[str]:
        norm = re.sub(r"\s+", " ", text.lower().strip())
        counts: Counter[str] = Counter()
        words = re.findall(r"[a-z0-9*!@#$]+", norm)
        for w in words:
            counts[f"w:{w}"] += 1
        for i in range(len(words) - 1):
            counts[f"w2:{words[i]}_{words[i + 1]}"] += 1

        padded = f"<{norm}>"
        n_len = len(padded)
        for k in range(self.char_min, self.char_max + 1):
            for i in range(max(0, n_len - k + 1)):
                counts[f"c:{padded[i : i + k]}"] += 1
        return counts

    def _vectorize(self, text: str) -> Dict[str, float]:
        counts = self._extract_raw_tokens(text)
        vec: Dict[str, float] = {}
        norm_sq = 0.0
        for feat, cnt in counts.items():
            idf = self._idf.get(feat)
            if idf is None:
                continue
            val = (1.0 + math.log(cnt)) * idf
            vec[feat] = val
            norm_sq += val * val
        if norm_sq > 0.0:
            inv_norm = 1.0 / math.sqrt(norm_sq)
            for feat in vec:
                vec[feat] *= inv_norm
        return vec

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        sample_weights: Sequence[float] | None = None,
    ) -> None:
        if not texts or len(texts) != len(labels):
            raise ValueError("texts and labels must be non-empty and equal length")
        n = len(texts)
        weights = list(sample_weights) if sample_weights is not None else [1.0] * n

        df: Counter[str] = Counter()
        raw_docs = [self._extract_raw_tokens(t) for t in texts]
        for doc in raw_docs:
            for feat in doc.keys():
                df[feat] += 1

        self._idf = {
            feat: math.log((1.0 + n) / (1.0 + count)) + 1.0 for feat, count in df.items()
        }
        self._weights = {feat: 0.0 for feat in self._idf}
        self._bias = 0.0

        vecs = [self._vectorize(t) for t in texts]
        for _ in range(self.epochs):
            grad_w: Dict[str, float] = {}
            grad_b = 0.0
            for vec, y, sw in zip(vecs, labels, weights):
                logit = self._bias + sum(self._weights.get(f, 0.0) * v for f, v in vec.items())
                pred = self._sigmoid(logit)
                err = (pred - float(y)) * sw
                grad_b += err
                for f, v in vec.items():
                    grad_w[f] = grad_w.get(f, 0.0) + err * v

            scale = self.learning_rate / n
            self._bias -= scale * grad_b
            for f in self._weights:
                g = grad_w.get(f, 0.0) / n + self.l2_reg * self._weights[f]
                self._weights[f] -= self.learning_rate * g

    def score_text(self, text: str, top_k: int = 5) -> ScoringResult:
        t0 = time.perf_counter()
        vec = self._vectorize(text)
        contributions: List[Tuple[str, float]] = []
        logit = self._bias
        for f, v in vec.items():
            w = self._weights.get(f, 0.0)
            contrib = w * v
            logit += contrib
            contributions.append((f, round(contrib, 5)))
        prob = self._sigmoid(logit)
        contributions.sort(key=lambda item: abs(item[1]), reverse=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ScoringResult(
            text=text,
            risk_probability=round(prob, 5),
            top_contributing_features=contributions[:top_k],
            latency_ms=round(elapsed_ms, 4),
        )

    def online_feedback_update(self, text: str, verified_label: int, weight: float = 2.0) -> float:
        if verified_label not in (0, 1):
            raise ValueError("verified_label must be 0 or 1")
        vec = self._vectorize(text)
        logit = self._bias + sum(self._weights.get(f, 0.0) * v for f, v in vec.items())
        pred = self._sigmoid(logit)
        err = (pred - float(verified_label)) * weight
        self._bias -= 0.25 * err
        for f, v in vec.items():
            self._weights[f] = self._weights.get(f, 0.0) - 0.25 * err * v
        return self.score_text(text).risk_probability
