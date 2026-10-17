"""Term-document scorers for keyword extraction experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite
import re
from typing import Protocol

import numpy as np
from scipy.stats import hypergeom
from scipy.special import expit, gammaln
from numpy.typing import NDArray


IntArray = NDArray[np.integer]
FloatArray = NDArray[np.float64]


_SCORER_LABELS = {
    "tf": "TF",
    "df": "DF",
    "cf": "CF",
    "subtf": "subTF",
    "tfidf": "TF-IDF",
    "subtfidf": "subTF-IDF",
    "btfidf": "BTF-IDF",
    "tficf": "TF-ICF",
    "bm25": "BM25",
    "g2": "G²",
    "simpleg2": "simple G²",
    "g2idf": "G²-IDF (DF-adjusted 2×2)",
    "poissondev": "Signed Poisson deviance",
    "poissonidf": "Poisson-IDF",
    "twopoisson": "Two-Poisson",
    "g2signed": "signed G²",
    "g2pos": "G²+",
    "g2neg": "G²−",
    "chi2": "χ²",
    "zscore": "z-score",
    "tscore": "t-score",
    "mi": "MI",
    "logdice": "logDice",
    "mi3": "MI³",
    "milogf": "MI.log-f",
    "minsens": "MinSens",
    "hgt": "HGT",
    "txm": "TXM",
    "txmfast": "TXM (recurrence)",
    "txmpos": "TXM+",
    "txmneg": "TXM−",
    "txmabs": "|TXM|",
    "extf": "extf",
    "logratio": "LogRatio",
    "simplemaths": "SimpleMaths",
}


def scorer_label(code: str) -> str:
    """Return the publication label for a canonical scorer code.

    Labels are plain Unicode text, not HTML or Matplotlib MathText. This keeps
    scorer typography independent of the rendering backend. Method names are
    upright text; lowercase symbols such as tf and alpha remain reserved for
    variables in formulas and prose.
    """
    code = code.strip().lower()
    fixed = _SCORER_LABELS.get(code)
    if fixed is not None:
        return fixed

    alpha_families = (
        ("g2idfa", "single-cell G²-IDF"),
        ("subtfidfa", "subTF-IDF"),
        ("tfidfa", "TF-IDF"),
        ("hgta", "HGT"),
        ("g2a", "G²"),
        ("chi2a", "χ²"),
    )
    for prefix, label in alpha_families:
        if code.startswith(prefix):
            value = code[len(prefix):]
            try:
                alpha = float(value)
            except ValueError as error:
                raise ValueError(f"Invalid scorer code: {code!r}") from error
            if not isfinite(alpha) or alpha < 0.0:
                raise ValueError(f"Invalid scorer code: {code!r}")
            return f"{label}α({alpha:g})"

    bm25_match = re.fullmatch(r"bm25k([0-9.]+)b([0-9.]+)", code)
    if bm25_match is not None:
        k1 = float(bm25_match.group(1))
        b = float(bm25_match.group(2))
        return f"BM25(k₁={k1:g},b={b:g})"

    if code.startswith("simplemathsk"):
        value = code[len("simplemathsk"):]
        try:
            k = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid scorer code: {code!r}") from error
        if not isfinite(k) or k <= 0.0:
            raise ValueError(f"Invalid scorer code: {code!r}")
        return f"SimpleMaths({k:g})"

    raise ValueError(f"Unknown scorer code: {code!r}")


class TermDocCorpus(Protocol):
    """Minimal corpus contract required by the scorers.

    Arrays use 1-based IDs: element 0 is unused. ``doc_len`` is the number of
    retained term occurrences in each document, ``cf`` is collection frequency,
    and ``df`` is document frequency.
    """

    n_docs: int
    collection_len: int
    avg_doc_len: float
    cf: NDArray[np.integer]
    df: NDArray[np.integer]
    doc_len: NDArray[np.integer]

    def tf(self, term_id: int, doc_id: int) -> int:
        """Return the frequency of one term in one document."""
        ...


class Scorer(ABC):
    """Base class for a term-document scorer."""

    def __init__(self, corpus: TermDocCorpus) -> None:
        self.corpus = corpus

    @property
    @abstractmethod
    def code(self) -> str:
        """Return the stable filename-safe scorer code."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Return a human-readable scorer name."""
        return self.__class__.__name__

    @property
    def label(self) -> str:
        """Return the publication label for this scorer."""
        return scorer_label(self.code)

    def score(self, term_id: int, doc_id: int) -> float:
        """Score one term/document cell.

        This scalar method is primarily for testing and inspection. Experiments
        should use :meth:`score_terms` to avoid Python-level loops.
        """
        tf = self.corpus.tf(term_id, doc_id)
        scores = self.score_terms(
            doc_id,
            np.asarray([term_id], dtype=np.int64),
            np.asarray([tf], dtype=np.int64),
        )
        return float(scores[0])

    def score_docs(
        self,
        term_id: int,
        doc_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score one term in several documents.

        The default implementation delegates to :meth:`score_terms` one
        document at a time. This is fast enough for interactive term queries;
        scorers can override it later if an experiment needs bulk term ->
        document scoring.
        """
        doc_ids = np.asarray(doc_ids, dtype=np.int64)
        tf = np.asarray(tf, dtype=np.int64)
        if doc_ids.shape != tf.shape:
            raise ValueError("doc_ids and tf must have the same shape")

        scores = np.empty(doc_ids.shape, dtype=np.float64)
        term_ids = np.asarray([term_id], dtype=np.int64)
        for i, (doc_id, count) in enumerate(zip(doc_ids, tf, strict=True)):
            scores[i] = self.score_terms(
                int(doc_id),
                term_ids,
                np.asarray([count], dtype=np.int64),
            )[0]
        return scores

    @abstractmethod
    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score aligned terms occurring in one document."""
        raise NotImplementedError


class Tf(Scorer):
    """Raw term frequency.

    tf

    tf : term frequency in document

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "tf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return raw term frequencies as floating-point scores."""
        del doc_id, term_ids
        return np.asarray(tf, dtype=np.float64)


class DocumentFrequency(Scorer):
    """Document frequency.

    df

    df : number of documents containing the term

    The score is corpus-level: for a term occurring in a focus document, the
    value is the same in every document of the corpus.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "df"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "DF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return document frequencies as floating-point scores."""
        del doc_id, tf
        return np.asarray(self.corpus.df[term_ids], dtype=np.float64)


class CollectionFrequency(Scorer):
    """Collection frequency.

    cf

    cf : number of occurrences of the term in the corpus

    The score is corpus-level: for a term occurring in a focus document, the
    value is the same in every document of the corpus.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "cf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "CF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return collection frequencies as floating-point scores."""
        del doc_id, tf
        return np.asarray(self.corpus.cf[term_ids], dtype=np.float64)


class SubTf(Scorer):
    """Sublinear term frequency.

    1 + ln(tf) if tf > 0, else 0

    tf : term frequency in document

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "subtf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "subTF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with sublinear term-frequency scaling."""
        del doc_id, term_ids
        observed = np.asarray(tf, dtype=np.float64)
        scores = np.zeros(observed.shape, dtype=np.float64)
        positive = observed > 0.0
        scores[positive] = 1.0 + np.log(observed[positive])
        return scores


class TfIdf(Scorer):
    """TF-IDF with raw term frequency.

    tf * ln(N / df)

    tf : term frequency in document
    N : number of documents
    df : document frequency of term

    No document-vector normalization is applied because it would not change the within-document term ranking.

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    def __init__(self, corpus: TermDocCorpus) -> None:
        super().__init__(corpus)
        self._idf = np.zeros(len(corpus.df), dtype=np.float64)
        valid = corpus.df > 0
        self._idf[valid] = np.log(corpus.n_docs / corpus.df[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "tfidf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TF-IDF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with TF-IDF."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        return observed * self._idf[term_ids]


class TfIdfAlpha(Scorer):
    """TF-IDF with a parametric IDF exponent.

    tf * ln(N / df)^alpha

    tf : term frequency in document
    N : number of documents
    df : document frequency of term
    alpha : strength of the inverse-document-frequency factor

    alpha=0 gives raw tf exactly. alpha=1 gives TF-IDF exactly. Increasing
    alpha increasingly favours terms with low document frequency; as alpha grows,
    terms with maximal IDF (df=1) dominate the ranking.

    This is an experimental parametric extension, not a standard named TF-IDF variant.

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    def __init__(self, corpus: TermDocCorpus, alpha: float = 1.0) -> None:
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        super().__init__(corpus)
        self.alpha = alpha
        self._idf = np.zeros(len(corpus.df), dtype=np.float64)
        valid = corpus.df > 0
        self._idf[valid] = np.log(corpus.n_docs / corpus.df[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return f"tfidfa{self.alpha:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"TF-IDF (IDF alpha={self.alpha:g})"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with TF-IDF at the configured IDF exponent."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        if self.alpha == 0.0:
            return observed.copy()
        return observed * np.power(self._idf[term_ids], self.alpha)


class BinaryTfIdf(Scorer):
    """Binary term frequency-inverse document frequency (BTF-IDF).

    1(tf > 0) * ln(N / df)

    tf : term frequency in document
    N : number of documents
    df : document frequency of term

    Ahmed, Z., Sheridan, P., McIsaac, M. & Farooque, A. A. (2026). "Common TF-IDF variants arise as key components in the test statistic of a penalized likelihood-ratio test for word burstiness." Discover Computing 29:274. doi:10.1007/s10791-026-10090-4.
    """

    def __init__(self, corpus: TermDocCorpus) -> None:
        super().__init__(corpus)
        self._idf = np.zeros(len(corpus.df), dtype=np.float64)
        valid = corpus.df > 0
        self._idf[valid] = np.log(corpus.n_docs / corpus.df[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "btfidf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "BTF-IDF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with binary TF-IDF."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        return (observed > 0.0).astype(np.float64) * self._idf[term_ids]


class TfIcf(Scorer):
    """Term frequency-inverse collection frequency (TF-ICF).

    tf * ln(CL / cf)

    tf : term frequency in document
    cf : collection frequency of term
    CL : collection length

    Kwok, K. L. (1990). "Experiments with a component theory of probabilistic information retrieval based on single terms as document components." ACM Transactions on Information Systems 8(4): 363-386.
    """

    def __init__(self, corpus: TermDocCorpus) -> None:
        if corpus.collection_len <= 0:
            raise ValueError("collection_len must be > 0")
        super().__init__(corpus)
        self._icf = np.zeros(len(corpus.cf), dtype=np.float64)
        valid = corpus.cf > 0
        self._icf[valid] = np.log(corpus.collection_len / corpus.cf[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "tficf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TF-ICF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with TF-ICF."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        return observed * self._icf[term_ids]


class SubTfIdf(Scorer):
    """TF-IDF with sublinear term-frequency scaling.

    (1 + ln(tf)) * ln(N / df) if tf > 0, else 0

    tf : term frequency in document
    N : number of documents
    df : document frequency of term

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    def __init__(self, corpus: TermDocCorpus) -> None:
        super().__init__(corpus)
        self._idf = np.zeros(len(corpus.df), dtype=np.float64)
        valid = corpus.df > 0
        self._idf[valid] = np.log(corpus.n_docs / corpus.df[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "subtfidf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "subTF-IDF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with sublinear TF-IDF."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        scores = np.zeros(observed.shape, dtype=np.float64)
        positive = observed > 0.0
        scores[positive] = (
            1.0 + np.log(observed[positive])
        ) * self._idf[term_ids[positive]]
        return scores


class SubTfIdfAlpha(Scorer):
    """Sublinear TF-IDF with a parametric IDF exponent.

    (1 + ln(tf)) * ln(N / df)^alpha if tf > 0, else 0

    tf : term frequency in document
    N : number of documents
    df : document frequency of term
    alpha : strength of the inverse-document-frequency factor

    alpha=0 gives subTF exactly. alpha=1 gives subTF-IDF exactly. Increasing
    alpha increasingly favours terms with low document frequency.

    Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management 24(5): 513-523. doi:10.1016/0306-4573(88)90021-0.
    """

    def __init__(self, corpus: TermDocCorpus, alpha: float = 1.0) -> None:
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        super().__init__(corpus)
        self.alpha = alpha
        self._idf = np.zeros(len(corpus.df), dtype=np.float64)
        valid = corpus.df > 0
        self._idf[valid] = np.log(corpus.n_docs / corpus.df[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return f"subtfidfa{self.alpha:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"subTF-IDF (IDF alpha={self.alpha:g})"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with sublinear TF-IDF at the configured IDF exponent."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        scores = np.zeros(observed.shape, dtype=np.float64)
        positive = observed > 0.0
        if not np.any(positive):
            return scores
        sub_tf = 1.0 + np.log(observed[positive])
        if self.alpha == 0.0:
            scores[positive] = sub_tf
        else:
            scores[positive] = sub_tf * np.power(
                self._idf[term_ids[positive]], self.alpha
            )
        return scores


class BM25(Scorer):
    """Lucene-style BM25 single-term contribution.

    idf = ln(1 + (N - df + 0.5) / (df + 0.5))
    K = k1 * (1 - b + b * dl / avgdl)
    idf * tf / (tf + K)

    tf : term frequency in document
    N : number of documents
    df : document frequency of term
    dl : document length
    avgdl : average document length
    k1 : BM25 term-frequency saturation parameter
    b : BM25 document-length normalization parameter

    The usual BM25 factor (k1 + 1) is constant for a fixed scorer and is omitted because it does not change within-document term ranking. Raw document lengths are used instead of Lucene's compact norm encoding.

    Lucene: org.apache.lucene.search.similarities.BM25Similarity.
    Robertson, S. & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in Information Retrieval 3(4): 333-389. doi:10.1561/1500000019.
    """

    def __init__(
        self,
        corpus: TermDocCorpus,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if not isfinite(k1) or k1 < 0.0:
            raise ValueError("k1 must be finite and >= 0")
        if not isfinite(b) or not 0.0 <= b <= 1.0:
            raise ValueError("b must be finite and in [0, 1]")
        if corpus.avg_doc_len <= 0.0:
            raise ValueError("avg_doc_len must be > 0")

        super().__init__(corpus)
        self.k1 = k1
        self.b = b

        df = np.asarray(corpus.df, dtype=np.float64)
        self._idf = np.zeros(df.shape, dtype=np.float64)
        valid = df > 0
        self._idf[valid] = np.log1p(
            (corpus.n_docs - df[valid] + 0.5) / (df[valid] + 0.5)
        )

    @property
    def code(self) -> str:
        """Return the filename code."""
        if self.k1 == 1.2 and self.b == 0.75:
            return "bm25"
        return f"bm25k{self.k1:g}b{self.b:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"Lucene BM25 (k1={self.k1:g}, b={self.b:g})"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with the Lucene BM25 single-term contribution."""
        tf_float = np.asarray(tf, dtype=np.float64)
        norm = self.k1 * (
            1.0
            - self.b
            + self.b
            * float(self.corpus.doc_len[doc_id])
            / float(self.corpus.avg_doc_len)
        )
        denominator = tf_float + norm
        scores = np.zeros(tf_float.shape, dtype=np.float64)
        positive = tf_float > 0
        scores[positive] = (
            self._idf[term_ids[positive]]
            * tf_float[positive]
            / denominator[positive]
        )
        return scores


def _term_doc_table(
    corpus: TermDocCorpus,
    doc_id: int,
    term_ids: IntArray,
    tf: IntArray,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    float,
    NDArray[np.bool_],
]:
    """Return the document-vs-rest 2 x 2 table for aligned terms.

    Cells are returned as:

        focus_term, other_term, focus_nonterm, other_nonterm

    followed by focus_tokens, other_tokens and an invalid-cell mask.
    """
    focus_term = np.asarray(tf, dtype=np.float64)
    focus_tokens = float(corpus.doc_len[doc_id])
    other_tokens = float(corpus.collection_len) - focus_tokens

    corpus_term = np.asarray(corpus.cf[term_ids], dtype=np.float64)
    other_term = corpus_term - focus_term
    focus_nonterm = focus_tokens - focus_term
    other_nonterm = other_tokens - other_term

    invalid = (
        (focus_term < 0.0)
        | (other_term < 0.0)
        | (focus_nonterm < 0.0)
        | (other_nonterm < 0.0)
    )
    return (
        focus_term,
        other_term,
        focus_nonterm,
        other_nonterm,
        focus_tokens,
        other_tokens,
        invalid,
    )


def _g2_values(
    corpus: TermDocCorpus,
    doc_id: int,
    term_ids: IntArray,
    tf: IntArray,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Return unsigned G², direction sign, and invalid mask for aligned terms."""
    (
        focus_term,
        other_term,
        focus_nonterm,
        other_nonterm,
        focus_tokens,
        other_tokens,
        invalid,
    ) = _term_doc_table(corpus, doc_id, term_ids, tf)

    if focus_tokens <= 0.0 or other_tokens <= 0.0:
        zeros = np.zeros(focus_term.shape, dtype=np.float64)
        return zeros, np.ones(focus_term.shape, dtype=np.float64), invalid

    all_tokens = focus_tokens + other_tokens
    all_term = focus_term + other_term
    all_nonterm = focus_nonterm + other_nonterm

    expected_focus_term = focus_tokens * all_term / all_tokens
    expected_other_term = other_tokens * all_term / all_tokens
    expected_focus_nonterm = focus_tokens * all_nonterm / all_tokens
    expected_other_nonterm = other_tokens * all_nonterm / all_tokens

    g2 = np.zeros(focus_term.shape, dtype=np.float64)
    for observed, expected in (
        (focus_term, expected_focus_term),
        (other_term, expected_other_term),
        (focus_nonterm, expected_focus_nonterm),
        (other_nonterm, expected_other_nonterm),
    ):
        valid = (observed > 0.0) & (expected > 0.0)
        g2[valid] += 2.0 * observed[valid] * np.log(observed[valid] / expected[valid])

    degenerate = (all_term == 0.0) | (all_nonterm == 0.0)
    g2[degenerate] = 0.0

    # G² is mathematically non-negative. Floating-point cancellation can
    # nevertheless produce tiny negative values near independence; clamp those
    # artifacts before fractional powers are applied by G2Alpha.
    finite = ~invalid & np.isfinite(g2)
    g2[finite] = np.maximum(g2[finite], 0.0)
    g2[invalid] = np.nan

    # Compare relative frequencies without division.
    sign = np.where(
        focus_term * other_tokens >= other_term * focus_tokens,
        1.0,
        -1.0,
    )
    sign[invalid] = np.nan
    return g2, sign, invalid


class simpleG2(Scorer):
    """Single-cell observed-versus-expected log-likelihood contribution.

    G2 = 2 * O * ln(O / E) if O > 0, else 0
    O = tf
    E = dl * cf / CL

    tf : observed term frequency in the document
    cf : total term frequency in the corpus
    dl : document length in tokens
    CL : corpus length in tokens

    E is the number of occurrences expected in a document of length dl if
    the term had the same relative frequency as in the whole corpus (cf / CL).

    This is one contribution to the full G² statistic, not a complete
    likelihood-ratio test. It is positive above expectation, negative below
    expectation, and zero at expectation. Compare with G2.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "simpleg2"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score observed frequencies against their corpus expectations."""
        observed = np.asarray(tf, dtype=np.float64)
        expected = (
            np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
            * float(self.corpus.doc_len[doc_id])
            / float(self.corpus.collection_len)
        )
        scores = np.zeros(observed.shape, dtype=np.float64)
        valid = (observed > 0.0) & (expected > 0.0)
        scores[valid] = 2.0 * observed[valid] * np.log(
            observed[valid] / expected[valid]
        )
        return scores


class G2(Scorer):
    """Full unsigned likelihood-ratio G² on a 2 x 2 document-vs-rest table.

    G2 = 2 * sum(O * ln(O / E)) over the four cells; O = 0 contributes 0.

    Observed counts O (rows: document, rest; columns: term, other terms):

                     term       other terms
        document     tf         dl - tf
        rest         cf - tf    CL - dl - cf + tf

    Each expected count E assumes independence between term and document:
    E = (row total * column total) / CL. Hence:

        E_document_term  = dl * cf / CL
        E_document_other = dl * (CL - cf) / CL
        E_rest_term      = (CL - dl) * cf / CL
        E_rest_other     = (CL - dl) * (CL - cf) / CL

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length in tokens
    CL : corpus length in tokens

    G² is non-negative. Direction is deliberately not encoded here; use
    SignedG2 when over- and under-representation must be distinguished.

    Dunning, T. (1993). "Accurate Methods for the Statistics of Surprise and Coincidence." Computational Linguistics 19(1): 61-74.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "g2"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return ordinary unsigned full G² values."""
        g2, _, _ = _g2_values(self.corpus, doc_id, term_ids, tf)
        return g2


def _residual_df_weights(corpus: TermDocCorpus) -> FloatArray:
    """Return expected-DF / observed-DF weights under random token placement.

    df_hat(t) = sum_j [1 - (1 - dl_j / CL)^cf(t)]

    Occurrences are independently placed across documents proportionally to
    their lengths. Terms with cf = df = 1 receive weight 1. Unused term IDs
    also receive weight 1, avoiding division by zero.
    """
    weights = np.ones(len(corpus.df), dtype=np.float64)
    valid = (corpus.cf > 0) & (corpus.df > 0)
    if not np.any(valid):
        return weights

    doc_proportions = (
        np.asarray(corpus.doc_len[1 : corpus.n_docs + 1], dtype=np.float64)
        / float(corpus.collection_len)
    )
    with np.errstate(divide="ignore"):
        log_nonoccurrence = np.log1p(-doc_proportions)

    frequencies, inverse = np.unique(corpus.cf[valid], return_inverse=True)
    expected_df = np.asarray(
        [-np.expm1(float(cf) * log_nonoccurrence).sum() for cf in frequencies],
        dtype=np.float64,
    )
    weights[valid] = expected_df[inverse] / corpus.df[valid]
    return weights


class G2IDF(Scorer):
    """Signed four-cell G² with a DF-adjusted background term frequency.

    Start with the document-vs-rest 2 x 2 table:

                          term          other terms
        target document   tf            dl - tf
        rest of corpus    b             (CL - dl) - b

    Instead of the actual outside term count c = cf - tf, use an *effective*
    count b, adjusted for how broadly the outside occurrences are distributed.
    For a term present in the target document, set r = df - 1. The expected
    number of other documents containing the remaining c occurrences is:

        r_hat = sum_(j != d) [1 - (1 - dl_j / (CL - dl))^c]

    With L = CL - dl, adjust the outside term odds by r / r_hat:

        b / (L - b) = (c / (L - c)) * (r / r_hat)
        b = L * c * r / ((L - c) * r_hat + c * r)

    If c = 0, set b = 0. For absent terms, use r = df instead of df - 1.
    The final score is the ordinary four-cell G² of this adjusted table,
    with the sign of the target-vs-background relative frequency difference.

    This is an experimental G²-shaped ranking score, not a likelihood-ratio
    test on observed token counts: b can be fractional and is not the true
    outside frequency. The expected DF uses independent allocation of the c
    remaining occurrences proportional to the *other* document lengths.

    cf = tf and df = 1 imply c = b = 0: a corpus singleton receives no
    additional DF bonus. There is no extra G² term, IDF exponent or weight.
    """

    def __init__(self, corpus: TermDocCorpus) -> None:
        super().__init__(corpus)
        self._lengths = np.asarray(
            corpus.doc_len[1 : corpus.n_docs + 1], dtype=np.float64
        )

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "g2idf"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return signed G² from a single, DF-adjusted 2 x 2 token table."""
        observed = np.asarray(tf, dtype=np.float64)
        corpus_frequency = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        document_frequency = np.asarray(self.corpus.df[term_ids], dtype=np.float64)
        dl = float(self.corpus.doc_len[doc_id])
        L = float(self.corpus.collection_len) - dl
        M = self.corpus.n_docs - 1

        scores = np.zeros(observed.shape, dtype=np.float64)
        if dl <= 0.0 or L <= 0.0 or M == 0:
            return scores

        c = corpus_frequency - observed
        present_here = (observed > 0.0).astype(np.float64)
        r = document_frequency - present_here
        invalid = (
            (observed < 0.0) | (observed > dl)
            | (c < 0.0) | (c > L)
            | (r < 0.0) | (r > M) | (r > c)
        )

        # For cf = tf, no occurrences remain outside the target document.
        # Else c >= 1 and r >= 1 for valid term/document counts.
        b = np.zeros(observed.shape, dtype=np.float64)
        valid = ~invalid & (c > 0.0)
        if np.any(valid):
            other_lengths = np.concatenate(
                (self._lengths[: doc_id - 1], self._lengths[doc_id:])
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                log_absent = np.log1p(-other_lengths / L)

            frequencies, inverse = np.unique(c[valid], return_inverse=True)
            expected_df = np.empty(len(frequencies), dtype=np.float64)
            # Chunk the outer product to bound peak memory use for long texts.
            for start in range(0, len(frequencies), 256):
                block = frequencies[start : start + 256]
                expected_df[start : start + len(block)] = (
                    -np.expm1(np.multiply.outer(block, log_absent))
                ).sum(axis=1)

            r_hat = expected_df[inverse]
            r_valid = r[valid]
            c_valid = c[valid]
            # Scale odds, keeping b + (L - b) == L. The 0/0 cases occur
            # only on invalid corpus statistics (or an empty outside corpus).
            denom = (L - c_valid) * r_hat + c_valid * r_valid
            with np.errstate(divide="ignore", invalid="ignore"):
                b[valid] = L * c_valid * r_valid / denom
            b[valid] = np.clip(b[valid], 0.0, L)
            invalid |= ~np.isfinite(b)

        # The adjusted outside term count is a token-equivalent count.
        # Recalculate all four expectations from the adjusted table margins.
        focus_nonterm = dl - observed
        other_nonterm = L - b
        term_total = observed + b
        nonterm_total = focus_nonterm + other_nonterm
        total = dl + L
        g2 = np.zeros(observed.shape, dtype=np.float64)
        for actual, expected in (
            (observed, dl * term_total / total),
            (b, L * term_total / total),
            (focus_nonterm, dl * nonterm_total / total),
            (other_nonterm, L * nonterm_total / total),
        ):
            positive = (actual > 0.0) & (expected > 0.0) & ~invalid
            g2[positive] += 2.0 * actual[positive] * np.log(
                actual[positive] / expected[positive]
            )

        g2 = np.maximum(g2, 0.0)
        sign = np.where(observed * L >= b * dl, 1.0, -1.0)
        scores = g2 * sign
        scores[(term_total == 0.0) | (nonterm_total == 0.0)] = 0.0
        scores[invalid] = np.nan
        return scores

class G2IDFAlpha(Scorer):
    """Single-cell G² contribution with a parametric IDF exponent.

    G2-IDFalpha = 2 * O * ln(O / E) * ln(N / df)^alpha, if O > 0;
                  otherwise 0.
    O = tf
    E = dl * cf / CL

    alpha : non-negative strength of the document-frequency factor

    alpha=0 gives simpleG2 exactly; alpha=1 gives single-cell G² times
    classical IDF, not the residual-DF weighting used by G2IDF.
    Under-represented terms (O < E) retain negative scores. The score
    uses one cell of the contingency table, not full four-cell G².
    """

    def __init__(self, corpus: TermDocCorpus, alpha: float = 1.0) -> None:
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        super().__init__(corpus)
        self.alpha = alpha
        self._idf = np.zeros(len(corpus.df), dtype=np.float64)
        valid = corpus.df > 0
        self._idf[valid] = np.log(corpus.n_docs / corpus.df[valid])

    @property
    def code(self) -> str:
        """Return the filename code."""
        return f"g2idfa{self.alpha:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"G²-IDF (IDF alpha={self.alpha:g})"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score observed frequencies against expectations and IDF^alpha."""
        observed = np.asarray(tf, dtype=np.float64)
        expected = (
            np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
            * float(self.corpus.doc_len[doc_id])
            / float(self.corpus.collection_len)
        )
        scores = np.zeros(observed.shape, dtype=np.float64)
        valid = (observed > 0.0) & (expected > 0.0)
        scores[valid] = 2.0 * observed[valid] * np.log(
            observed[valid] / expected[valid]
        )
        if self.alpha != 0.0:
            scores[valid] *= np.power(self._idf[term_ids[valid]], self.alpha)
        return scores


class SignedPoissonDeviance(Scorer):
    """Signed Poisson deviance for observed versus expected term counts.

    D = sign(O - E) * 2 * (O * ln(O / E) - O + E)
    O = tf
    E = dl * cf / CL

    tf : observed term frequency in the document
    cf : collection frequency of the term
    dl : document length in tokens
    CL : corpus length in tokens

    The O * ln(O / E) term is defined as 0 when O = 0, giving -2 * E
    for a term absent from the document (E > 0). The signed deviance is
    positive above expectation, negative below, and zero at expectation.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "poissondev"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return the signed Poisson deviance for each term."""
        observed = np.asarray(tf, dtype=np.float64)
        expected = (
            np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
            * float(self.corpus.doc_len[doc_id])
            / float(self.corpus.collection_len)
        )
        deviance = np.zeros(observed.shape, dtype=np.float64)
        valid = expected > 0.0
        deviance[valid] = expected[valid]  # O = 0: O * ln(O/E) = 0
        positive = valid & (observed > 0.0)
        deviance[positive] += (
            observed[positive] * np.log(observed[positive] / expected[positive])
            - observed[positive]
        )
        # The deviance is non-negative; avoid negative round-off at O ~= E.
        deviance[valid] = np.maximum(deviance[valid], 0.0)
        return 2.0 * np.sign(observed - expected) * deviance


class PoissonIDF(SignedPoissonDeviance):
    """Signed Poisson deviance weighted by residual document concentration.

    PoissonIDF = sign(O - E) * 2 * [O * ln(O / E) - O + E] * df_hat / df
    O = tf
    E = dl * cf / CL

    df_hat = sum_j [1 - (1 - dl_j / CL)^cf], over all corpus documents.

    df_hat is the expected number of documents containing a term if its cf
    occurrences are independently placed according to document lengths.
    The ratio df_hat / df rewards concentration beyond what cf explains.
    In particular, cf = df = 1 implies df_hat = 1 and multiplier = 1.

    This is an experimental residual-IDF weighting, not classical IDF.
    """

    def __init__(self, corpus: TermDocCorpus) -> None:
        super().__init__(corpus)
        self._df_weight = _residual_df_weights(corpus)

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "poissonidf"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Multiply signed deviance by the precomputed residual-DF factor."""
        return super().score_terms(doc_id, term_ids, tf) * self._df_weight[term_ids]


class TwoPoisson(Scorer):
    """Length-aware two-Poisson mixture for topical versus background usage.

    For each term, document counts X_j are fitted by maximum likelihood:

        X_j ~ (1 - pi) * Pois(lambda_0 * dl_j / avg_dl)
                + pi  * Pois(lambda_1 * dl_j / avg_dl)

    with 0 < lambda_0 < lambda_1 and pi the topical-document prior.
    Parameters are estimated by EM from *all* document counts, including zeros.
    The score is the log posterior odds of the topical component:

        log(pi / (1 - pi)) + tf * log(lambda_1 / lambda_0)
        - (lambda_1 - lambda_0) * dl / avg_dl

    This retains a useful ranking range when probabilities saturate near 1.
    Document frequency is reflected in the zero counts in the likelihood,
    rather than introduced as a separate IDF multiplier. This is a heuristic
    keyword score based on a fitted mixture, not a document likelihood ratio.

    A two-component fit is accepted only if its BIC improves on the single-
    Poisson fit (two additional parameters); otherwise the term receives zero
    score. That model-selection rule avoids spurious topical components for
    insufficiently supported words, especially single occurrences.

    Fits are cached per term. On first use, term counts are obtained through
    corpus.tf(term_id, doc_id) for every document. Optionally, a corpus can
    provide postings_for_term(term_id) -> (doc_ids, frequencies), reducing the
    time needed to retrieve the nonzero counts. No additional dependency is
    required by the existing TermDocCorpus interface.
    """

    def __init__(self, corpus: TermDocCorpus) -> None:
        super().__init__(corpus)
        if corpus.avg_doc_len <= 0.0 or corpus.n_docs <= 0:
            raise ValueError("TwoPoisson requires a nonempty corpus")
        self._exposure = np.asarray(
            corpus.doc_len[1:corpus.n_docs + 1], dtype=np.float64
        ) / corpus.avg_doc_len
        self._valid_docs = self._exposure > 0.0
        self._exposure_valid = self._exposure[self._valid_docs]
        self._n_valid = int(np.count_nonzero(self._valid_docs))
        count = len(corpus.cf)
        self._fitted = np.zeros(count, dtype=np.bool_)
        self._log_prior = np.zeros(count, dtype=np.float64)
        self._log_rate_ratio = np.zeros(count, dtype=np.float64)
        self._rate_difference = np.zeros(count, dtype=np.float64)

    @property
    def code(self) -> str:
        return "twopoisson"

    @property
    def name(self) -> str:
        return "Two-Poisson topical/background mixture"

    def _fit_term(self, term_id: int) -> None:
        """Fit a two-component Poisson mixture once, using EM and BIC."""
        if self._fitted[term_id]:
            return
        self._fitted[term_id] = True
        if self._n_valid < 3 or int(self.corpus.cf[term_id]) < 2:
            return

        # An optional term-oriented accessor avoids N scalar tf() lookups.
        get_postings = getattr(self.corpus, "postings_for_term", None)
        if callable(get_postings):
            doc_ids, frequencies = get_postings(term_id)
            counts = np.zeros(self.corpus.n_docs, dtype=np.float64)
            ids = np.asarray(doc_ids, dtype=np.int64)
            counts[ids - 1] = np.asarray(frequencies, dtype=np.float64)
        else:
            counts = np.fromiter(
                (self.corpus.tf(term_id, doc_id)
                 for doc_id in range(1, self.corpus.n_docs + 1)),
                dtype=np.float64, count=self.corpus.n_docs,
            )
        y = counts[self._valid_docs]
        x = self._exposure_valid
        if np.any(y < 0.0) or np.any(~np.isfinite(y)):
            raise ValueError(f"Invalid term frequencies for term ID {term_id}")
        total = float(y.sum())
        if total < 2.0:
            return
        mean = total / float(x.sum())
        nonzero = int(np.count_nonzero(y))

        # Start with rare topical documents, a low background intensity, and
        # a higher topical intensity. All rates are measured per avg document.
        pi = float(np.clip(0.7 * nonzero / self._n_valid,
                           1.0 / self._n_valid, 0.6))
        lam0 = max(mean * 0.1, 1e-9)
        lam1 = max((mean - (1.0 - pi) * lam0) / pi, lam0 * 1.01)

        for _ in range(80):
            odds = (np.log(pi / (1.0 - pi))
                    + y * np.log(lam1 / lam0) - (lam1 - lam0) * x)
            q = expit(odds)
            pi_new = float(np.clip(np.mean(q), 1e-9, 1.0 - 1e-9))
            lam0_new = max(float(np.dot(1.0 - q, y) / np.dot(1.0 - q, x)), 1e-9)
            lam1_new = max(float(np.dot(q, y) / np.dot(q, x)), 1e-9)
            if lam0_new > lam1_new:
                lam0_new, lam1_new = lam1_new, lam0_new
                pi_new = 1.0 - pi_new
            change = max(
                abs(pi_new - pi),
                abs(np.log(lam0_new / lam0)),
                abs(np.log(lam1_new / lam1)),
            )
            pi, lam0, lam1 = pi_new, lam0_new, lam1_new
            if change < 1e-6:
                break

        # Compare mixture and single Poisson likelihoods; factorial terms
        # cancel. A fitted mixture has 3 parameters versus 1 for the null.
        background = y * np.log(lam0 * x) - lam0 * x
        topical = y * np.log(lam1 * x) - lam1 * x
        ll_mixture = np.logaddexp(
            np.log1p(-pi) + background,
            np.log(pi) + topical,
        ).sum()
        ll_single = np.sum(y * np.log(mean * x) - mean * x)
        bic_gain = 2.0 * (ll_mixture - ll_single) - 2.0 * np.log(self._n_valid)
        if bic_gain <= 0.0 or lam1 <= lam0 * (1.0 + 1e-6):
            return

        self._log_prior[term_id] = np.log(pi / (1.0 - pi))
        self._log_rate_ratio[term_id] = np.log(lam1 / lam0)
        self._rate_difference[term_id] = lam1 - lam0

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return topical-component posterior log odds for aligned terms."""
        ids = np.asarray(term_ids, dtype=np.int64)
        observed = np.asarray(tf, dtype=np.float64)
        if ids.shape != observed.shape:
            raise ValueError("term_ids and tf must have the same shape")
        for term_id in np.unique(ids[~self._fitted[ids]]):
            self._fit_term(int(term_id))
        exposure = float(self.corpus.doc_len[doc_id]) / self.corpus.avg_doc_len
        scores = (
            self._log_prior[ids] + observed * self._log_rate_ratio[ids]
            - exposure * self._rate_difference[ids]
        )
        # Absent terms are not keyword candidates.
        scores[observed <= 0.0] = 0.0
        # If EM does not justify a mixture, zero is a neutral score.
        return scores


class SignedG2(Scorer):
    """Directional G² association score.

    The magnitude is ordinary G². The sign is positive when the term relative
    frequency is at least as high in the focus document as in the rest of the
    corpus, otherwise negative.

    This is a signed association score built from G²; G² itself remains
    intrinsically non-negative.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "g2signed"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Signed G²"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return G² with the direction of association attached."""
        g2, sign, _ = _g2_values(self.corpus, doc_id, term_ids, tf)
        return g2 * sign


class G2Pos(Scorer):
    """Positive directional G² only: over-representation magnitude.

    This is the positive part of SignedG2. Under-represented terms are set to 0.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "g2pos"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Positive G²"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Keep only over-represented G² magnitudes."""
        g2, sign, _ = _g2_values(self.corpus, doc_id, term_ids, tf)
        scores = np.zeros(g2.shape, dtype=np.float64)
        positive = sign > 0.0
        scores[positive] = g2[positive]
        scores[np.isnan(g2) | np.isnan(sign)] = np.nan
        return scores


class G2Neg(Scorer):
    """Negative directional G² only: under-representation magnitude.

    Under-represented terms keep their G² magnitude as a positive value;
    over-represented terms are set to 0.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "g2neg"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Negative G²"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Keep only under-represented G² magnitudes."""
        g2, sign, _ = _g2_values(self.corpus, doc_id, term_ids, tf)
        scores = np.zeros(g2.shape, dtype=np.float64)
        negative = sign < 0.0
        scores[negative] = g2[negative]
        scores[np.isnan(g2) | np.isnan(sign)] = np.nan
        return scores


class ExclusiveTf(Scorer):
    """Raw term frequency restricted to terms exclusive to the focus document.

    tf if tf = cf, otherwise 0

    tf : term frequency in document
    cf : collection frequency of term

    This is a deliberately simple exclusivity baseline: selection is based only
    on exclusivity, while ranking among exclusive terms remains raw frequency.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "extf"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Exclusive TF"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return tf only when all corpus occurrences are in this document."""
        del doc_id
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        scores = np.zeros(observed.shape, dtype=np.float64)
        exclusive = (observed > 0.0) & (observed == corpus_term)
        scores[exclusive] = observed[exclusive]
        return scores



class G2Alpha(Scorer):
    """Parametric G² specificity.

    G2alpha = tf * (G2 / tf)^alpha
            = tf^(1-alpha) * G2^alpha

    tf : observed frequency in the focus
    G2 : unsigned likelihood-ratio statistic on the 2 x 2 table
    alpha : strength of the G²-per-occurrence specificity factor

    alpha=0 gives raw tf exactly. alpha=1 gives ordinary G² exactly. Values
    above 1 continue the same family by increasingly favouring terms with high
    G² per observed occurrence. No document-frequency statistic is required; the
    scorer depends only on the 2 x 2 contingency table.

    G² is unsigned. Consequently, for alpha>1 a very strong under-representation
    can also receive a large score. Use G2Pos when only positive association is
    desired.

    Dunning, T. (1993). "Accurate Methods for the Statistics of Surprise and Coincidence." Computational Linguistics 19(1): 61-74.
    """

    def __init__(
        self,
        corpus: TermDocCorpus,
        alpha: float = 1.0,
    ) -> None:
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        super().__init__(corpus)
        self.alpha = alpha

    @property
    def code(self) -> str:
        """Return the filename code."""
        return f"g2a{self.alpha:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"G² alpha={self.alpha:g}"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with G² alpha at the configured alpha."""
        observed = np.asarray(tf, dtype=np.float64)

        if self.alpha == 0.0:
            return observed.copy()

        g2, _, invalid = _g2_values(self.corpus, doc_id, term_ids, tf)
        if self.alpha == 1.0:
            return g2

        scores = np.zeros(g2.shape, dtype=np.float64)
        valid = (observed > 0.0) & (g2 > 0.0) & ~invalid
        specificity = np.divide(
            g2[valid],
            observed[valid],
        )
        scores[valid] = observed[valid] * np.power(specificity, self.alpha)
        scores[invalid] = np.nan
        return scores


def _chi2_values(
    corpus: TermDocCorpus,
    doc_id: int,
    term_ids: IntArray,
    tf: IntArray,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Return unsigned Pearson X², direction sign, and invalid mask."""
    (
        focus_term,
        other_term,
        focus_nonterm,
        other_nonterm,
        focus_tokens,
        other_tokens,
        invalid,
    ) = _term_doc_table(corpus, doc_id, term_ids, tf)

    if focus_tokens <= 0.0 or other_tokens <= 0.0:
        zeros = np.zeros(focus_term.shape, dtype=np.float64)
        return zeros, np.ones(focus_term.shape, dtype=np.float64), invalid

    all_tokens = focus_tokens + other_tokens
    all_term = focus_term + other_term
    all_nonterm = focus_nonterm + other_nonterm

    expected_focus_term = focus_tokens * all_term / all_tokens
    expected_other_term = other_tokens * all_term / all_tokens
    expected_focus_nonterm = focus_tokens * all_nonterm / all_tokens
    expected_other_nonterm = other_tokens * all_nonterm / all_tokens

    chi2 = np.zeros(focus_term.shape, dtype=np.float64)
    for observed, expected in (
        (focus_term, expected_focus_term),
        (other_term, expected_other_term),
        (focus_nonterm, expected_focus_nonterm),
        (other_nonterm, expected_other_nonterm),
    ):
        valid = expected > 0.0
        delta = observed[valid] - expected[valid]
        chi2[valid] += delta * delta / expected[valid]

    sign = np.where(
        focus_term * other_tokens >= other_term * focus_tokens,
        1.0,
        -1.0,
    )
    chi2[invalid] = np.nan
    sign[invalid] = np.nan
    return chi2, sign, invalid


class Chi2(Scorer):
    """Signed Pearson chi-square on a 2 x 2 term/document table.

    X2 = sum((O - E)^2 / E)

    The same 2 x 2 table as G² is used. The sign is positive when the term
    relative frequency is at least as high in the focus document as in the rest
    of the collection, otherwise negative.

    Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in the case of correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling." Philosophical Magazine 50(302): 157-175. doi:10.1080/14786440009463897.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "chi2"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with signed Pearson X²."""
        chi2, sign, _ = _chi2_values(self.corpus, doc_id, term_ids, tf)
        return chi2 * sign


class Chi2Alpha(Scorer):
    """Parametric positive Pearson chi-square specificity.

    Chi2Alpha = tf * (X2 / tf)^alpha

    X2 : unsigned Pearson chi-square magnitude on the 2 x 2 table
    tf : observed frequency in the focus
    alpha : strength of chi-square per observed occurrence

    alpha=0 gives raw tf exactly. For alpha>0, only positively associated
    terms are retained. alpha=1 therefore gives the positive branch of ordinary
    signed chi-square exactly. Increasing alpha increasingly favours terms with
    high positive chi-square per observed occurrence.

    Pearson, K. (1900). "On the criterion that a given system of deviations from the probable in the case of correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling." Philosophical Magazine 50(302): 157-175. doi:10.1080/14786440009463897.
    """

    def __init__(self, corpus: TermDocCorpus, alpha: float = 1.0) -> None:
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        super().__init__(corpus)
        self.alpha = alpha

    @property
    def code(self) -> str:
        """Return the filename code."""
        return f"chi2a{self.alpha:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"χ² alpha={self.alpha:g}"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with chi-square alpha at the configured alpha."""
        observed = np.asarray(tf, dtype=np.float64)
        if self.alpha == 0.0:
            return observed.copy()

        chi2, sign, invalid = _chi2_values(self.corpus, doc_id, term_ids, tf)
        scores = np.zeros(chi2.shape, dtype=np.float64)
        positive = (sign > 0.0) & ~invalid

        if self.alpha == 1.0:
            scores[positive] = chi2[positive]
            scores[invalid] = np.nan
            return scores

        valid = positive & (observed > 0.0) & (chi2 > 0.0)
        specificity = np.divide(chi2[valid], observed[valid])
        scores[valid] = observed[valid] * np.power(specificity, self.alpha)
        scores[invalid] = np.nan
        return scores


class ZScore(Scorer):
    """Hypergeometric z-score for a term in the focus document.

    expected = cf * dl / CL
    variance = dl * (cf / CL) * (1 - cf / CL) * (CL - dl) / (CL - 1)
    z = (tf - expected) / sqrt(variance)

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    Positive values indicate over-representation and negative values indicate
    under-representation. With fixed 2 x 2 margins, z^2 differs from Pearson X^2
    only by the finite-population factor (CL - 1) / CL.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "zscore"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "z-score"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with the hypergeometric z-score."""
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])
        corpus_size = float(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (corpus_size <= 0.0)
            | (part_size > corpus_size)
            | (corpus_term > corpus_size)
        )
        if corpus_size <= 1.0 or part_size <= 0.0:
            scores[invalid] = np.nan
            return scores

        probability = corpus_term / corpus_size
        expected = corpus_term * part_size / corpus_size
        variance = (
            part_size
            * probability
            * (1.0 - probability)
            * (corpus_size - part_size)
            / (corpus_size - 1.0)
        )
        valid = (variance > 0.0) & ~invalid
        scores[valid] = (observed[valid] - expected[valid]) / np.sqrt(variance[valid])
        scores[invalid] = np.nan
        return scores


class TScore(Scorer):
    """Corpus-linguistic t-score association measure.

    expected = cf * dl / CL
    t = (tf - expected) / sqrt(tf)

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    This is the conventional collocation t-score heuristic, not Student's
    t-test. Positive values indicate over-representation. Its sqrt(tf)
    denominator makes it retain more preference for repeated evidence than MI.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "tscore"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "t-score"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with corpus-linguistic t-score."""
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])
        corpus_size = float(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (corpus_size <= 0.0)
            | (part_size > corpus_size)
            | (corpus_term > corpus_size)
        )
        if corpus_size <= 0.0 or part_size <= 0.0:
            scores[invalid] = np.nan
            return scores

        expected = corpus_term * part_size / corpus_size
        valid = (observed > 0.0) & ~invalid
        scores[valid] = (observed[valid] - expected[valid]) / np.sqrt(observed[valid])
        scores[invalid] = np.nan
        return scores


class MutualInformation(Scorer):
    """Corpus-linguistic Mutual Information (MI) score.

    MI = log2((tf / dl) / (cf / CL))
       = log2(tf * CL / (cf * dl))

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    This measure is conventionally called MI in corpus-linguistic collocation
    interfaces, including Frantext. In information theory the cell-level score
    is Pointwise Mutual Information (PMI); full mutual information is the
    probability-weighted sum over all cells of the joint distribution.

    Positive values indicate a term rate above its collection expectation;
    0 indicates independence at that cell; negative values indicate a lower
    rate. Base 2 follows the conventional MI/PMI presentation and does not
    affect ranking.

    Church, K. W. & Hanks, P. (1990). "Word Association Norms, Mutual Information, and Lexicography." Computational Linguistics 16(1): 22-29.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "mi"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "Mutual Information (MI)"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with corpus-linguistic MI (PMI)."""
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])
        corpus_size = float(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (corpus_size <= 0.0)
            | (part_size > corpus_size)
            | (corpus_term > corpus_size)
        )
        if corpus_size <= 0.0 or part_size <= 0.0:
            scores[invalid] = np.nan
            return scores

        valid = (observed > 0.0) & (corpus_term > 0.0) & ~invalid
        scores[valid] = np.log2(
            observed[valid] * corpus_size / (corpus_term[valid] * part_size)
        )
        scores[invalid] = np.nan
        return scores


class LogDice(Scorer):
    """Sketch Engine logDice association score.

    Dice = 2 * tf / (cf + dl)
    logDice = 14 + log2(Dice)

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length

    The term is treated as one member of the association and the focus
    document as the other: tf is their co-occurrence count, cf is the term
    marginal, and dl is the document marginal. The additive constant 14 is the
    conventional Sketch Engine scale shift and does not affect ranking.

    Rychlý, P. (2008). "A Lexicographer-Friendly Association Score." RASLAN 2008: 6-9.
    """

    @property
    def code(self) -> str:
        return "logdice"

    @property
    def name(self) -> str:
        return "logDice"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (observed > part_size)
        )
        denominator = corpus_term + part_size
        valid = (observed > 0.0) & (denominator > 0.0) & ~invalid
        scores[valid] = 14.0 + np.log2(
            2.0 * observed[valid] / denominator[valid]
        )
        scores[invalid] = np.nan
        return scores


class MutualInformation3(Scorer):
    """Sketch Engine MI3 association score.

    MI3 = log2(tf^3 * CL / (cf * dl))

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    Relative to MI, MI3 adds 2 * log2(tf), reducing MI's preference for
    low-frequency events.

    Oakes, M. P. (1998). Statistics for Corpus Linguistics. Edinburgh University Press.
    """

    @property
    def code(self) -> str:
        return "mi3"

    @property
    def name(self) -> str:
        return "MI³"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])
        corpus_size = float(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (corpus_size <= 0.0)
            | (part_size > corpus_size)
            | (corpus_term > corpus_size)
        )
        if corpus_size <= 0.0 or part_size <= 0.0:
            scores[invalid] = np.nan
            return scores

        valid = (observed > 0.0) & (corpus_term > 0.0) & ~invalid
        scores[valid] = (
            np.log2(
                observed[valid] * corpus_size
                / (corpus_term[valid] * part_size)
            )
            + 2.0 * np.log2(observed[valid])
        )
        scores[invalid] = np.nan
        return scores


class MutualInformationLogFrequency(Scorer):
    """Sketch Engine MI.log-f association score (formerly salience).

    MI.log-f = MI * ln(tf + 1)
    MI = log2(tf * CL / (cf * dl))

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    The MI factor uses base 2 and the frequency multiplier the natural
    logarithm, following the Sketch Engine definition.

    Kilgarriff, A., Rychlý, P., Smrž, P. & Tugwell, D. (2004). "The Sketch Engine." Proceedings of EURALEX 2004: 105-116.
    """

    @property
    def code(self) -> str:
        return "milogf"

    @property
    def name(self) -> str:
        return "MI.log-f"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])
        corpus_size = float(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (corpus_size <= 0.0)
            | (part_size > corpus_size)
            | (corpus_term > corpus_size)
        )
        if corpus_size <= 0.0 or part_size <= 0.0:
            scores[invalid] = np.nan
            return scores

        valid = (observed > 0.0) & (corpus_term > 0.0) & ~invalid
        mi = np.log2(
            observed[valid] * corpus_size
            / (corpus_term[valid] * part_size)
        )
        scores[valid] = mi * np.log1p(observed[valid])
        scores[invalid] = np.nan
        return scores


class MinimumSensitivity(Scorer):
    """Sketch Engine minimum-sensitivity association score.

    minimum sensitivity = min(tf / cf, tf / dl)

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length

    Pedersen, T. (1998). "Dependent Bigram Identification." Proceedings of AAAI-98: 428-433.
    """

    @property
    def code(self) -> str:
        return "minsens"

    @property
    def name(self) -> str:
        return "Minimum sensitivity"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        observed = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        part_size = float(self.corpus.doc_len[doc_id])

        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = (
            (observed < 0.0)
            | (corpus_term < observed)
            | (part_size < 0.0)
            | (observed > part_size)
        )
        valid = (
            (observed > 0.0)
            & (corpus_term > 0.0)
            & (part_size > 0.0)
            & ~invalid
        )
        scores[valid] = np.minimum(
            observed[valid] / corpus_term[valid],
            observed[valid] / part_size,
        )
        scores[invalid] = np.nan
        return scores


class Hgt(Scorer):
    """One-sided hypergeometric-test term weight (HGT).

    HGT = -ln(P(X >= tf))
    X ~ Hypergeom(CL, cf, dl)

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    The natural logarithm follows the TF-IDF comparison used by Sheridan and
    Onsjö. This scorer measures over-representation only and is non-negative.

    Sheridan, P. & Onsjö, M. (2024). "The hypergeometric test performs comparably to TF-IDF on standard text analysis tasks." Multimedia Tools and Applications 83: 28875-28890. doi:10.1007/s11042-023-16615-z.
    """

    MAX_SCORE = 1000.0 * np.log(10.0)

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "hgt"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "HGT"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms by upper-tail hypergeometric surprisal."""
        observed = np.asarray(tf, dtype=np.int64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.int64)
        part_size = int(self.corpus.doc_len[doc_id])
        corpus_size = int(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        if part_size <= 0 or corpus_size <= 0 or part_size > corpus_size:
            return scores

        invalid = (
            (observed < 0)
            | (corpus_term < observed)
            | (corpus_term > corpus_size)
            | (observed > part_size)
        )
        valid = (observed > 0) & ~invalid
        if np.any(valid):
            # scipy.stats.hypergeom.logsf is much slower than sf for vector
            # inputs. Compute the ordinary survival probability first and only
            # fall back to logsf for values that underflow to zero.
            probability = hypergeom.sf(
                observed[valid] - 1,
                corpus_size,
                corpus_term[valid],
                part_size,
            )
            values = np.empty(probability.shape, dtype=np.float64)
            finite = probability > 0.0
            values[finite] = -np.log(probability[finite])
            if np.any(~finite):
                values[~finite] = -hypergeom.logsf(
                    observed[valid][~finite] - 1,
                    corpus_size,
                    corpus_term[valid][~finite],
                    part_size,
                )
            values[~np.isfinite(values)] = self.MAX_SCORE
            scores[valid] = values

        scores[invalid] = np.nan
        return scores


class HgtAlpha(Scorer):
    """Parametric extension of the positive HGT score.

    HgtAlpha = tf * (HGT / tf)^alpha
    HGT = -ln(P(X >= tf))

    tf : term frequency in document
    alpha : strength of HGT surprisal per observed occurrence

    alpha=0 gives raw tf exactly. alpha=1 gives HGT exactly. This is an
    experimental parametric extension, not a standard named weighting scheme.
    """

    def __init__(self, corpus: TermDocCorpus, alpha: float = 1.0) -> None:
        if not isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        super().__init__(corpus)
        self.alpha = alpha
        self._hgt = Hgt(corpus)

    @property
    def code(self) -> str:
        """Return the filename code."""
        return f"hgta{self.alpha:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"HGT alpha={self.alpha:g}"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with HGT alpha at the configured alpha."""
        observed = np.asarray(tf, dtype=np.float64)
        if self.alpha == 0.0:
            return observed.copy()

        hgt = self._hgt.score_terms(doc_id, term_ids, tf)
        scores = np.zeros(observed.shape, dtype=np.float64)
        invalid = np.isnan(hgt)
        positive = (observed > 0.0) & (hgt > 0.0) & ~invalid
        scores[positive] = observed[positive] * np.power(
            hgt[positive] / observed[positive],
            self.alpha,
        )
        scores[invalid] = np.nan
        return scores


class Txm(Scorer):
    """Signed TXM/Lafon lexical specificity.

    X ~ Hypergeom(CL, cf, dl)
    mode = floor((cf + 1) * (dl + 1) / (CL + 2))

    -log10(P(X >= tf)) if tf > mode
     log10(P(X <= tf)) if tf < mode
     0                         if tf == mode

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    TXM determines over- versus under-representation from the mode of the
    hypergeometric distribution, not from its mean. Floating-point log10 scores
    are retained for numerical experiments; display formatting is left to the
    caller.

    Lafon, P. (1980). "Sur la variabilité de la fréquence des formes dans un corpus." Mots 1: 127-165. doi:10.3406/mots.1980.1008.
    TXM User Manual, section "Spécificités".
    """

    MAX_SCORE = 1000.0

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "txm"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TXM"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with signed TXM/Lafon hypergeometric specificity."""
        observed = np.asarray(tf, dtype=np.int64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.int64)
        part_size = int(self.corpus.doc_len[doc_id])
        corpus_size = int(self.corpus.collection_len)

        scores = np.zeros(observed.shape, dtype=np.float64)
        if part_size <= 0 or corpus_size <= 0 or part_size > corpus_size:
            return scores

        invalid = (
            (observed < 0)
            | (corpus_term < observed)
            | (corpus_term > corpus_size)
            | (observed > part_size)
        )

        mode = np.floor(
            (corpus_term.astype(np.float64) + 1.0)
            * (part_size + 1.0)
            / (corpus_size + 2.0)
        ).astype(np.int64)

        positive = (observed > mode) & ~invalid
        if np.any(positive):
            probability = hypergeom.sf(
                observed[positive] - 1,
                corpus_size,
                corpus_term[positive],
                part_size,
            )
            values = np.empty(probability.shape, dtype=np.float64)
            finite = probability > 0.0
            values[finite] = -np.log10(probability[finite])
            if np.any(~finite):
                values[~finite] = -hypergeom.logsf(
                    observed[positive][~finite] - 1,
                    corpus_size,
                    corpus_term[positive][~finite],
                    part_size,
                ) / np.log(10.0)
            values[~np.isfinite(values)] = self.MAX_SCORE
            scores[positive] = values

        negative = (observed < mode) & ~invalid
        if np.any(negative):
            probability = hypergeom.cdf(
                observed[negative],
                corpus_size,
                corpus_term[negative],
                part_size,
            )
            values = np.empty(probability.shape, dtype=np.float64)
            finite = probability > 0.0
            values[finite] = np.log10(probability[finite])
            if np.any(~finite):
                values[~finite] = hypergeom.logcdf(
                    observed[negative][~finite],
                    corpus_size,
                    corpus_term[negative][~finite],
                    part_size,
                ) / np.log(10.0)
            values[~np.isfinite(values)] = -self.MAX_SCORE
            scores[negative] = values

        scores[invalid] = np.nan
        return scores


class TxmFast(Scorer):
    """Signed TXM specificity using a logarithmic hypergeometric tail recurrence.

    Uses the same hypergeometric model, mode-based sign, and base-10 scores
    as :class:`Txm`. The calculation follows Camacho-Collados, Pilehvar &
    Navigli (2016), Section 3.1.1: start from log P(X=tf), accumulate ratios
    P(X=k+1)/P(X=k) (or the reverse ratios for the lower tail), and stop
    when the remaining tail is negligible. There is no vocabulary filtering
    or score threshold.

    Camacho-Collados, J., Pilehvar, M. T. & Navigli, R. (2016).
    "Nasari: Integrating explicit knowledge and corpus statistics for a
    multilingual representation of concepts and entities." Artificial
    Intelligence 240: 36-64. doi:10.1016/j.artint.2016.07.005.
    """

    MAX_SCORE = 1000.0
    # Truncation targets 1e-12 relative error in each probability tail.
    TAIL_RTOL = 1e-12

    @property
    def code(self) -> str:
        return "txmfast"

    @property
    def name(self) -> str:
        return "TXM recurrence"

    @staticmethod
    def _log_tail(
        counts: IntArray,
        corpus_term: IntArray,
        doc_len: int,
        collection_len: int,
        upper: bool,
    ) -> FloatArray:
        """Compute log P(X>=k) or log P(X<=k) without underflow.

        All input arrays are aligned and valid. Active tails advance together
        with NumPy operations; the only Python loop is over tail *depth*, not
        individual term/document pairs.
        """
        k = np.asarray(counts, dtype=np.float64)
        K = np.asarray(corpus_term, dtype=np.float64)
        n = float(doc_len)
        M = float(collection_len)

        # log C(K,k) + log C(M-K,n-k) - log C(M,n)
        log_pmf = (
            gammaln(K + 1.0) - gammaln(k + 1.0)
            - gammaln(K - k + 1.0)
            + gammaln(M - K + 1.0) - gammaln(n - k + 1.0)
            - gammaln(M - K - n + k + 1.0)
            - (gammaln(M + 1.0) - gammaln(n + 1.0)
               - gammaln(M - n + 1.0))
        )

        endpoint = np.minimum(K, n) if upper else np.maximum(0.0, n - (M - K))
        current = k.copy()
        term = np.ones(k.shape, dtype=np.float64)
        total = np.ones(k.shape, dtype=np.float64)
        active = current < endpoint if upper else current > endpoint
        while np.any(active):
            a = np.flatnonzero(active)
            position = current[a]
            if upper:
                ratio = ((K[a] - position) * (n - position)) / (
                    (position + 1.0) * (M - K[a] - n + position + 1.0)
                )
                current[a] += 1.0
            else:
                ratio = (position * (M - K[a] - n + position)) / (
                    (K[a] - position + 1.0) * (n - position + 1.0)
                )
                current[a] -= 1.0

            term[a] *= ratio
            total[a] += term[a]
            # Past the mode, recurrence ratios decrease. The omitted tail is
            # bounded geometrically by term * ratio/(1-ratio). A conservative
            # bound avoids stopping too early for broad hypergeom tails.
            bound = np.zeros_like(ratio)
            decreasing = ratio < 1.0
            bound[decreasing] = (
                term[a][decreasing] * ratio[decreasing]
                / (1.0 - ratio[decreasing])
            )
            # Continue until the estimated remaining mass is small, or until
            # the support endpoint is reached. The bound is based on the last
            # recurrence ratio, which is an upper bound on subsequent ratios.
            active[a] = (
                ((current[a] < endpoint[a]) if upper else (current[a] > endpoint[a]))
                & (~decreasing | (bound > TxmFast.TAIL_RTOL * total[a]))
            )

        return log_pmf + np.log(total)

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms using signed hypergeometric tail surprisal."""
        observed = np.asarray(tf, dtype=np.int64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.int64)
        part_size = int(self.corpus.doc_len[doc_id])
        corpus_size = int(self.corpus.collection_len)
        scores = np.zeros(observed.shape, dtype=np.float64)
        if part_size <= 0 or corpus_size <= 0 or part_size > corpus_size:
            return scores
        invalid = (
            (observed < 0) | (corpus_term < observed)
            | (corpus_term > corpus_size) | (observed > part_size)
        )
        mode = np.floor(
            (corpus_term.astype(np.float64) + 1.0)
            * (part_size + 1.0) / (corpus_size + 2.0)
        ).astype(np.int64)
        positive = (observed > mode) & ~invalid
        if np.any(positive):
            lp = self._log_tail(observed[positive], corpus_term[positive],
                                part_size, corpus_size, upper=True)
            scores[positive] = np.minimum(-lp / np.log(10.0), self.MAX_SCORE)
        negative = (observed < mode) & ~invalid
        if np.any(negative):
            lp = self._log_tail(observed[negative], corpus_term[negative],
                                part_size, corpus_size, upper=False)
            scores[negative] = np.maximum(lp / np.log(10.0), -self.MAX_SCORE)
        scores[invalid] = np.nan
        return scores


class TxmPos(Txm):
    """Positive TXM specificity only: over-representation magnitude."""

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "txmpos"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TXM positive"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Keep only positive TXM scores."""
        return np.maximum(super().score_terms(doc_id, term_ids, tf), 0.0)


class TxmNeg(Txm):
    """Negative TXM specificity only, returned as positive magnitudes."""

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "txmneg"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TXM negative"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Keep only under-representation magnitudes."""
        return np.maximum(-super().score_terms(doc_id, term_ids, tf), 0.0)


class TxmAbs(Txm):
    """Absolute TXM specificity: significance magnitude without direction."""

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "txmabs"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return "TXM absolute"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Return the absolute magnitude of signed TXM specificity."""
        return np.abs(super().score_terms(doc_id, term_ids, tf))


class LogRatio(Scorer):
    """Hardie's Log Ratio.

    log2((tf / dl) / ((cf - tf) / (CL - dl)))

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length

    Log Ratio is the binary logarithm of the ratio of relative frequencies. A zero count is replaced by 0.5 to avoid an infinite ratio.

    Hardie, A. (2014). "Log Ratio – an informal introduction." ESRC Centre for Corpus Approaches to Social Science (CASS), Lancaster University, 28 April 2014.
    """

    @property
    def code(self) -> str:
        """Return the filename code."""
        return "logratio"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with Hardie's Log Ratio."""
        tf_float = np.asarray(tf, dtype=np.float64)
        dl = float(self.corpus.doc_len[doc_id])
        CL = float(self.corpus.collection_len)
        cf = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)

        rest_len = CL - dl
        if dl <= 0.0 or rest_len <= 0.0:
            return np.zeros(tf_float.shape, dtype=np.float64)

        rest_tf = cf - tf_float
        invalid = (tf_float < 0.0) | (rest_tf < 0.0)

        # Hardie's standard Log Ratio compares relative frequencies. Replacing
        # a zero observed count by 0.5 keeps the ratio finite.
        doc_count = np.where(tf_float > 0.0, tf_float, 0.5)
        rest_count = np.where(rest_tf > 0.0, rest_tf, 0.5)

        scores = np.log2(
            (doc_count / dl)
            / (rest_count / rest_len)
        )
        scores[invalid] = np.nan
        return scores


class SimpleMaths(Scorer):
    """Kilgarriff Simple Maths.

    rf_doc = 1_000_000 * tf / dl
    rf_rest = 1_000_000 * (cf - tf) / (CL - dl)
    (rf_doc + k) / (rf_rest + k)

    tf : term frequency in document
    cf : collection frequency of term
    dl : document length
    CL : collection length
    k : smoothing parameter, per million ("N" in Kilgarriff 2009)

    k chooses the frequency band of the keywords: small k favours rare
    terms, large k common ones. Kilgarriff's values (1, 10, 100, 1000) are
    set for corpus-vs-corpus comparison, where a focus corpus has millions of
    words. In a document of dl tokens a single occurrence is already
    1_000_000 / dl per million (about 234 for a 4,276-token chapter), so
    k = 1 leaves every term absent from the rest at a score proportional to
    its tf: the ranking degenerates into a list of hapaxes tied at tf = 1.

    The default is therefore expressed in document units: half an
    occurrence in a document of average length,
    k = 500_000 / avg_doc_len. A single occurrence of a term absent from the
    rest then scores about 3, so terms with repeated evidence rank first.
    Pass k explicitly to use a fixed per-million value.

    Kilgarriff, A. (2009). "Simple Maths for Keywords." Proceedings of the Corpus Linguistics Conference CL2009, University of Liverpool.
    """

    def __init__(self, corpus: TermDocCorpus, k: float | None = None) -> None:
        if k is None:
            if corpus.avg_doc_len <= 0.0:
                raise ValueError("avg_doc_len must be > 0 for the default k")
            k = 500_000.0 / float(corpus.avg_doc_len)
            self.auto_k = True
        else:
            self.auto_k = False
        if not isfinite(k) or k <= 0.0:
            raise ValueError("k must be finite and > 0")
        super().__init__(corpus)
        self.k = k

    @property
    def code(self) -> str:
        """Return the filename code."""
        if self.auto_k:
            return "simplemaths"
        return f"simplemathsk{self.k:g}"

    @property
    def name(self) -> str:
        """Return the human-readable scorer name."""
        return f"Simple Maths (k={self.k:g})"

    def score_terms(
        self,
        doc_id: int,
        term_ids: IntArray,
        tf: IntArray,
    ) -> FloatArray:
        """Score terms with Simple Maths."""
        focus_tokens = float(self.corpus.doc_len[doc_id])
        other_tokens = float(self.corpus.collection_len) - focus_tokens
        if focus_tokens <= 0.0 or other_tokens <= 0.0:
            return np.zeros(len(tf), dtype=np.float64)

        focus_term = np.asarray(tf, dtype=np.float64)
        corpus_term = np.asarray(self.corpus.cf[term_ids], dtype=np.float64)
        other_term = corpus_term - focus_term
        ppm_focus = focus_term * 1_000_000.0 / focus_tokens + self.k
        ppm_other = other_term * 1_000_000.0 / other_tokens + self.k
        return ppm_focus / ppm_other


SCORER_TYPES: tuple[type[Scorer], ...] = (
    BinaryTfIdf,
    BM25,
    Chi2,
    Chi2Alpha,
    CollectionFrequency,
    DocumentFrequency,
    ExclusiveTf,
    G2,
    G2IDF,
    G2IDFAlpha,
    simpleG2,
    G2Alpha,
    G2Neg,
    G2Pos,
    Hgt,
    HgtAlpha,
    LogDice,
    LogRatio,
    MinimumSensitivity,
    MutualInformation,
    MutualInformation3,
    MutualInformationLogFrequency,
    SignedG2,
    SignedPoissonDeviance,
    PoissonIDF,
    TwoPoisson,
    SimpleMaths,
    SubTf,
    SubTfIdf,
    SubTfIdfAlpha,
    Tf,
    TfIcf,
    TfIdf,
    TfIdfAlpha,
    TScore,
    Txm,
    TxmFast,
    TxmAbs,
    TxmNeg,
    TxmPos,
    ZScore,
)


def default_scorers(corpus: TermDocCorpus) -> tuple[Scorer, ...]:
    """Return the scorer configurations used by the current keyword experiment."""
    return (
        BinaryTfIdf(corpus),
        BM25(corpus, 100.0, 1.0),
        BM25(corpus),
        Chi2(corpus),
        Chi2Alpha(corpus, 00.0),
        Chi2Alpha(corpus, 00.05),
        Chi2Alpha(corpus, 00.15),
        Chi2Alpha(corpus, 00.25),
        Chi2Alpha(corpus, 00.35),
        Chi2Alpha(corpus, 00.43),
        Chi2Alpha(corpus, 00.5),
        Chi2Alpha(corpus, 00.75),
        Chi2Alpha(corpus, 01.0),
        Chi2Alpha(corpus, 02.0),
        Chi2Alpha(corpus, 04.0),
        Chi2Alpha(corpus, 08.0),
        Chi2Alpha(corpus, 16.0),
        CollectionFrequency(corpus),
        DocumentFrequency(corpus),
        ExclusiveTf(corpus),
        G2(corpus),
        simpleG2(corpus),
        G2IDF(corpus),
        G2IDFAlpha(corpus, 0.0),
        G2IDFAlpha(corpus, 0.05),
        G2IDFAlpha(corpus, 0.1),
        G2IDFAlpha(corpus, 0.15),
        G2IDFAlpha(corpus, 0.2),
        G2IDFAlpha(corpus, 0.25),
        G2IDFAlpha(corpus, 0.3),
        G2IDFAlpha(corpus, 0.5),
        G2IDFAlpha(corpus, 0.75),
        G2IDFAlpha(corpus, 1.0),
        G2IDFAlpha(corpus, 2.0),
        G2IDFAlpha(corpus, 4.0),
        G2IDFAlpha(corpus, 8.0),
        G2IDFAlpha(corpus, 16.0),
        SignedPoissonDeviance(corpus),
        PoissonIDF(corpus),
        TwoPoisson(corpus),
        G2Alpha(corpus, 00.0),
        G2Alpha(corpus, 00.25),
        G2Alpha(corpus, 00.5),
        G2Alpha(corpus, 00.75),
        G2Alpha(corpus, 01.0),
        G2Alpha(corpus, 01.4),
        G2Alpha(corpus, 01.7),
        G2Alpha(corpus, 02.0),
        G2Alpha(corpus, 04.0),
        G2Alpha(corpus, 08.0),
        G2Alpha(corpus, 16.0),
        Hgt(corpus),
        HgtAlpha(corpus, 00.0),
        HgtAlpha(corpus, 00.25),
        HgtAlpha(corpus, 00.5),
        HgtAlpha(corpus, 00.75),
        HgtAlpha(corpus, 01.0),
        HgtAlpha(corpus, 01.47),
        HgtAlpha(corpus, 02.0),
        HgtAlpha(corpus, 04.0),
        HgtAlpha(corpus, 08.0),
        HgtAlpha(corpus, 16.0),
        LogDice(corpus),
        LogRatio(corpus),
        MinimumSensitivity(corpus),
        MutualInformation(corpus),
        MutualInformation3(corpus),
        MutualInformationLogFrequency(corpus),
        SimpleMaths(corpus, 1.0),
        SimpleMaths(corpus),
        SubTf(corpus),
        SubTfIdf(corpus),
        SubTfIdfAlpha(corpus, 00.0),
        SubTfIdfAlpha(corpus, 00.05),
        SubTfIdfAlpha(corpus, 00.1),
        SubTfIdfAlpha(corpus, 00.15),
        SubTfIdfAlpha(corpus, 00.2),
        SubTfIdfAlpha(corpus, 00.25),
        SubTfIdfAlpha(corpus, 00.56),
        SubTfIdfAlpha(corpus, 00.75),
        SubTfIdfAlpha(corpus, 01.0),
        SubTfIdfAlpha(corpus, 02.0),
        SubTfIdfAlpha(corpus, 04.0),
        SubTfIdfAlpha(corpus, 08.0),
        SubTfIdfAlpha(corpus, 16.0),
        Tf(corpus),
        TfIcf(corpus),
        TfIdf(corpus),
        TfIdfAlpha(corpus, 00.0),
        TfIdfAlpha(corpus, 00.25),
        TfIdfAlpha(corpus, 00.5),
        TfIdfAlpha(corpus, 00.75),
        TfIdfAlpha(corpus, 01.14),
        TfIdfAlpha(corpus, 01.5),
        TfIdfAlpha(corpus, 02.0),
        TfIdfAlpha(corpus, 04.0),
        TfIdfAlpha(corpus, 08.0),
        TfIdfAlpha(corpus, 16.0),
        TScore(corpus),
        Txm(corpus),
        TxmFast(corpus),
        ZScore(corpus),
    )


def make_scorer(corpus: TermDocCorpus, code: str) -> Scorer:
    """Create a scorer from its stable experiment code.

    Canonical codes include ``tf``, ``df``, ``cf``, ``subtf``, ``tfidf``, ``subtfidf``,
    ``btfidf``, ``tficf``, ``hgt``, ``txm``, ``txmfast``, ``bm25``, ``g2``, ``simpleg2``, ``g2idf``,
    ``poissondev``, ``poissonidf``, ``twopoisson``, ``chi2``,
    ``zscore``, ``tscore``, ``mi``, ``logdice``, ``mi3``, ``milogf``,
    ``minsens``, ``extf``, ``logratio`` and ``simplemaths``.

    Parametric families use an explicit numeric suffix: ``tfidfaA``,
    ``subtfidfaA``, ``g2idfaA``, ``hgtaA``, ``g2aA`` and ``chi2aA``.
    """
    code = code.strip().lower()
    factories = {
        "tf": Tf,
        "df": DocumentFrequency,
        "cf": CollectionFrequency,
        "subtf": SubTf,
        "tfidf": TfIdf,
        "subtfidf": SubTfIdf,
        "btfidf": BinaryTfIdf,
        "tficf": TfIcf,
        "hgt": Hgt,
        "txm": Txm,
        "txmfast": TxmFast,
        "txmpos": TxmPos,
        "txmneg": TxmNeg,
        "txmabs": TxmAbs,
        "bm25": BM25,
        "g2": G2,
        "simpleg2": simpleG2,
        "g2idf": G2IDF,
        "poissondev": SignedPoissonDeviance,
        "poissonidf": PoissonIDF,
        "twopoisson": TwoPoisson,
        "g2signed": SignedG2,
        "g2pos": G2Pos,
        "g2neg": G2Neg,
        "chi2": Chi2,
        "zscore": ZScore,
        "tscore": TScore,
        "mi": MutualInformation,
        "logdice": LogDice,
        "mi3": MutualInformation3,
        "milogf": MutualInformationLogFrequency,
        "minsens": MinimumSensitivity,
        "extf": ExclusiveTf,
        "logratio": LogRatio,
        "simplemaths": SimpleMaths,
    }
    factory = factories.get(code)
    if factory is not None:
        return factory(corpus)

    if code.startswith("subtfidfa"):
        value = code[len("subtfidfa"):]
        try:
            alpha = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid subTF-IDF alpha scorer code: {code!r}") from error
        return SubTfIdfAlpha(corpus, alpha)

    if code.startswith("tfidfa"):
        value = code[len("tfidfa"):]
        try:
            alpha = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid TF-IDF alpha scorer code: {code!r}") from error
        return TfIdfAlpha(corpus, alpha)

    if code.startswith("g2idfa"):
        value = code[len("g2idfa"):]
        try:
            alpha = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid G²-IDF alpha scorer code: {code!r}") from error
        return G2IDFAlpha(corpus, alpha)

    if code.startswith("hgta"):
        value = code[len("hgta"):]
        try:
            alpha = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid HGT alpha scorer code: {code!r}") from error
        return HgtAlpha(corpus, alpha)

    if code.startswith("g2a"):
        value = code[len("g2a"):]
        try:
            alpha = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid G² alpha scorer code: {code!r}") from error
        return G2Alpha(corpus, alpha)

    if code.startswith("chi2a"):
        value = code[len("chi2a"):]
        try:
            alpha = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid χ² alpha scorer code: {code!r}") from error
        return Chi2Alpha(corpus, alpha)

    bm25_match = re.fullmatch(r"bm25k([0-9.]+)b([0-9.]+)", code)
    if bm25_match is not None:
        return BM25(corpus, float(bm25_match.group(1)), float(bm25_match.group(2)))

    if code.startswith("simplemathsk"):
        value = code[len("simplemathsk"):]
        try:
            k = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid Simple Maths scorer code: {code!r}") from error
        return SimpleMaths(corpus, k)

    known = ", ".join(
        (
            *factories.keys(),
            "bm25k100b1",
            "simplemathsk1",
            "tfidfa0.5",
            "subtfidfa0.5",
            "g2idfa0.3",
            "hgta0.5",
            "g2a2",
            "chi2a2",
        )
    )
    raise ValueError(f"Unknown scorer {code!r}. Known scorer codes: {known}")
