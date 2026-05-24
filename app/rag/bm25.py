"""
BM25 sparse encoder.

Produces (indices, values) pairs compatible with Qdrant's SparseVector format.
Uses rank_bm25 under the hood; vocabulary is built lazily on first encode call
and can be refreshed by calling build_index() explicitly.

The encoder is intentionally stateless across requests — each instance owns
its own vocabulary so it can be used independently for curriculum vs memory.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import structlog

logger = structlog.get_logger(__name__)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "it",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "but",
        "not",
        "with",
        "as",
        "by",
        "from",
        "that",
        "this",
        "was",
        "are",
        "be",
        "been",
        "have",
        "has",
        "do",
        "does",
        "will",
        "would",
        "can",
        "could",
        "should",
        "its",
        "their",
        "there",
        "then",
        "than",
        "so",
        "if",
        "about",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25Encoder:
    """
    Lightweight BM25 encoder that converts text to sparse vectors.

    Parameters
    ----------
    k1 : float  — term saturation (default 1.5)
    b  : float  — length normalization (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._vocab: dict[str, int] = {}  # token → int index
        self._idf: dict[int, float] = {}  # token_idx → IDF score
        self._avgdl: float = 0.0
        self._corpus_size: int = 0
        self._fitted: bool = False

    def fit(self, corpus: list[str]) -> BM25Encoder:
        """
        Build vocabulary and IDF scores from a corpus of documents.
        Must be called before encode().
        """
        tokenized = [_tokenize(doc) for doc in corpus]
        self._corpus_size = len(tokenized)
        self._avgdl = (
            sum(len(t) for t in tokenized) / self._corpus_size if self._corpus_size else 1.0
        )

        # Assign integer indices to unique tokens
        all_tokens = {tok for doc in tokenized for tok in doc}
        self._vocab = {tok: idx for idx, tok in enumerate(sorted(all_tokens))}

        # Compute IDF for each token
        doc_freq: Counter = Counter()
        for doc_tokens in tokenized:
            for tok in set(doc_tokens):
                doc_freq[tok] += 1

        self._idf = {}
        for tok, idx in self._vocab.items():
            df = doc_freq.get(tok, 0)
            # Robertson-Sparck Jones IDF with smoothing
            self._idf[idx] = math.log((self._corpus_size - df + 0.5) / (df + 0.5) + 1)

        self._fitted = True
        logger.info(
            "bm25_fitted",
            vocab_size=len(self._vocab),
            corpus_size=self._corpus_size,
        )
        return self

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        """
        Encode a document for indexing (full BM25 score).
        Returns (indices, values) for Qdrant SparseVector.
        """
        self._assert_fitted()
        tokens = _tokenize(text)
        dl = len(tokens)
        tf = Counter(tokens)
        indices: list[int] = []
        values: list[float] = []

        for tok, count in tf.items():
            idx = self._vocab.get(tok)
            if idx is None:
                continue  # OOV token — skip
            idf = self._idf[idx]
            numerator = count * (self.k1 + 1)
            denominator = count + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score = idf * (numerator / denominator)
            indices.append(idx)
            values.append(score)

        return indices, values

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """
        Encode a query for retrieval (IDF-only weights, no length norm).
        Returns (indices, values) for Qdrant SparseVector.

        Design note: IDF is a document-frequency weight, not a term count.
        Repeated query tokens do NOT accumulate extra IDF weight — each
        unique token contributes its IDF score exactly once.
        """
        self._assert_fitted()
        tokens = _tokenize(text)
        seen: dict[int, float] = {}

        for tok in tokens:
            idx = self._vocab.get(tok)
            if idx is None:
                continue
            # Use setdefault — first occurrence wins; duplicates are ignored.
            seen.setdefault(idx, self._idf[idx])

        indices = list(seen.keys())
        values = list(seen.values())
        return indices, values

    def vocab_size(self) -> int:
        return len(self._vocab)

    # ── private ─────────────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "BM25Encoder must be fitted before encoding. " "Call encoder.fit(corpus) first."
            )
