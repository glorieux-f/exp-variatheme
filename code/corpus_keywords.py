"""Rank corpus-wide keywords from PREFIX{docs,terms,postings}.tsv.

Local scorers come from scorers.py. Their document-level scores can be
aggregated independently with:
  sum       sum of signed local scores
  tf-l2     sqrt(sum(tf * max(local_score, 0)**2))
  power     signed p-norm sum (default p=1.2)
  satloss   sum of signed local scores × BM25 dispersion loss

Without --aggregators, reproduce the previous default behavior:
  bm25 -> sum; all other local scorers -> tf-l2.
The corpus-level ``kl`` (sqrt(CF) × KL), ``js`` (sqrt(CF) ×
Jensen–Shannon), and ``jslog`` (log1p(CF/tau) × Jensen–Shannon)
scorers need no aggregation. ``--families`` shows automatically inferred
term neighborhoods using document-level excess co-occurrence.

Vocabulary selection is a post-ranking filter: all scorer statistics always
refer to the complete original corpus.
Requires NumPy and SciPy (through scorers.py).
"""

import argparse
import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from scorers import BM25, make_scorer


VOCABS = ("all", "content", "nostops", "nocaps", "stops", "caps", "stopscaps")
DEFAULT_STOPWORDS = Path(__file__).with_name("stopwords.txt")
STOPWORD_VOCABS = {"content", "nostops", "stops", "stopscaps"}


@dataclass
class Corpus:
    """Minimal corpus interface expected by scorers.py."""

    n_docs: int
    doc_ids: np.ndarray
    collection_len: int
    avg_doc_len: float
    cf: np.ndarray
    df: np.ndarray
    doc_len: np.ndarray
    posting_keys: np.ndarray
    posting_tf: np.ndarray

    def tf(self, term_id: int, doc_id: int) -> int:
        """Return one term/document frequency for scalar scorer inspection."""
        key = term_id * len(self.doc_len) + doc_id
        pos = int(np.searchsorted(self.posting_keys, key))
        if pos < len(self.posting_keys) and self.posting_keys[pos] == key:
            return int(self.posting_tf[pos])
        return 0


def rows(path):
    with path.open(encoding="utf-8", newline="") as source:
        yield from csv.DictReader(source, delimiter="\t")


def normalize_word(word):
    """Normalize a lemma for stopword membership checks."""
    return unicodedata.normalize("NFC", word.strip()).casefold()


def load_stopwords(path):
    """Read a UTF-8 word list, matching keywords-glob.py semantics."""
    words = set()
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            word = line.strip()
            if word and not word.startswith("#") and word != "__STOPWORDS":
                words.add(normalize_word(word))
    return words


def vocabulary_mask(lemmas, stopwords, mode):
    """Filter lemma forms without changing document or corpus statistics."""
    if mode == "all":
        return np.ones(len(lemmas), dtype=bool)
    caps = np.fromiter((bool(word[:1].isupper()) for word in lemmas),
                       dtype=bool, count=len(lemmas))
    stops = np.fromiter((normalize_word(word) in stopwords for word in lemmas),
                        dtype=bool, count=len(lemmas))
    return {
        "content": ~caps & ~stops,
        "nostops": ~stops,
        "nocaps": ~caps,
        "stops": stops,
        "caps": caps,
        "stopscaps": stops | caps,
    }[mode]


def load(prefix):
    """Read and validate the complete contingency table."""
    docs = list(rows(Path(f"{prefix}docs.tsv")))
    terms = list(rows(Path(f"{prefix}terms.tsv")))
    lengths = np.zeros(max(int(row["doc_id"]) for row in docs) + 1, dtype=np.int64)
    for row in docs:
        lengths[int(row["doc_id"])] = int(row["doc_len"])

    ids = np.array([int(row["term_id"]) for row in terms], dtype=np.int64)
    lemmas = [""] * (int(ids.max()) + 1)
    cf = np.zeros(len(lemmas), dtype=np.int64)
    df = np.zeros(len(lemmas), dtype=np.int64)
    for row in terms:
        term_id = int(row["term_id"])
        lemmas[term_id] = row["lemma"]
        cf[term_id] = int(row["cf"])
        df[term_id] = int(row["df"])

    tid, did, tf = np.loadtxt(
        f"{prefix}postings.tsv", delimiter="\t", skiprows=1, dtype=np.int64
    ).T
    D = int(np.count_nonzero(lengths))
    N = int(lengths.sum())
    if not np.array_equal(np.bincount(tid, minlength=len(cf)), df):
        raise ValueError("DF totals disagree with postings")
    if not np.array_equal(np.bincount(tid, weights=tf, minlength=len(cf)), cf):
        raise ValueError("CF totals disagree with postings")
    if not np.array_equal(np.bincount(did, weights=tf, minlength=len(lengths)), lengths):
        raise ValueError("Document lengths disagree with postings")
    if np.any(tf <= 0) or np.any(lengths[did] <= 0) or np.any(cf[tid] < tf):
        raise ValueError("Invalid document-term posting")

    # Term-major ordered postings permit scalar lookups without a dense matrix.
    keys = tid * len(lengths) + did
    if np.any(keys[1:] <= keys[:-1]):
        raise ValueError("Postings must be ordered by term_id and doc_id")
    corpus = Corpus(D, np.flatnonzero(lengths), N, N / D, cf, df, lengths, keys, tf)
    return corpus, ids, lemmas, tid, did, tf


def kl(corpus, tid, did, tf):
    """Original corpus-level sqrt(CF) * KL scorer."""
    cf = corpus.cf
    contribution = tf * np.log(tf * corpus.collection_len / (cf[tid] * corpus.doc_len[did]))
    total = np.bincount(tid, weights=contribution, minlength=len(cf))
    scores = np.zeros(len(cf), dtype=np.float64)
    valid = cf > 0
    scores[valid] = np.maximum(0, total[valid] / np.sqrt(cf[valid]))
    return scores


def js_divergence(corpus, tid, did, tf):
    """Corpus-wide Jensen–Shannon divergence, in natural logs.

    Compare q_d = tf_d / CF with p_d = length_d / collection_length.
    Avoid a dense term-document array by accounting for absent terms using
    the identity JS(q,p) = ln(2) - 1/2 Σ_{q_d>0} [q_d ln(1+p_d/q_d)
                                               + p_d ln(1+q_d/p_d)].
    """
    cf = corpus.cf
    q = tf / cf[tid]
    p = corpus.doc_len[did] / corpus.collection_len
    correction = q * np.log1p(p / q) + p * np.log1p(q / p)
    total = np.bincount(tid, weights=correction, minlength=len(cf))
    scores = np.zeros(len(cf), dtype=np.float64)
    valid = cf > 0
    scores[valid] = np.maximum(0.0, np.log(2.0) - 0.5 * total[valid])
    return scores


def js(corpus, tid, did, tf):
    """Original sqrt(CF) × Jensen–Shannon score."""
    return np.sqrt(corpus.cf) * js_divergence(corpus, tid, did, tf)


def jslog(corpus, tid, did, tf, tau):
    """Jensen–Shannon weighted by log(1 + CF/tau), with tau > 0."""
    return np.log1p(corpus.cf / tau) * js_divergence(corpus, tid, did, tf)


def make_family_profiles(corpus, tid, did, tf):
    """Build normalized term vectors of positive length-adjusted excess use."""
    expected = corpus.cf[tid] * corpus.doc_len[did] / corpus.collection_len
    weights = np.sqrt(tf) * np.maximum(0.0, np.log((tf + 0.5) / (expected + 0.5)))
    matrix = sparse.csr_matrix(
        (weights, (tid, did)), shape=(len(corpus.cf), len(corpus.doc_len))
    )
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inverse = np.zeros(len(norms))
    np.divide(1.0, norms, out=inverse, where=norms > 0)
    return sparse.diags(inverse) @ matrix


def term_neighborhoods(corpus, matrix, seeds, mask, size, minimum_similarity):
    """Find nearest terms by cosine similarity of the excess-use profiles."""
    # Exclude terms with little evidence and very ubiquitous vocabulary.
    candidates = (mask & (corpus.cf >= 20) & (corpus.df >= 10)
                  & (corpus.df <= 0.8 * corpus.n_docs))
    families = []
    for seed in seeds:
        similarities = (matrix @ matrix.getrow(int(seed)).T).toarray().ravel()
        similarities[~candidates] = -1.0
        similarities[seed] = -1.0
        ranked = np.argsort(-similarities, kind="stable")[:max(size - 1, 0)]
        neighbors = [(int(j), float(similarities[j])) for j in ranked
                     if similarities[j] >= minimum_similarity]
        families.append((int(seed), neighbors))
    return families


def write_families(path, families, lemmas):
    """Write one seed and its nearest associated terms on each line."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for seed, neighbors in families:
            words = ", ".join(lemmas[term_id] for term_id, _ in neighbors)
            stream.write(f"{lemmas[seed]}: {words}\n" if words
                         else f"{lemmas[seed]}\n")


def write_family_edges(path, families, lemmas):
    """Export the cosine associations underlying the readable family lists."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("rank", "seed_id", "seed", "neighbor_id", "neighbor", "cosine"))
        for rank, (seed, neighbors) in enumerate(families, 1):
            for neighbor, similarity in neighbors:
                writer.writerow((rank, seed, lemmas[seed], neighbor, lemmas[neighbor],
                                 f"{similarity:.6f}"))


def bm25_saturation_loss(corpus, tid, did, tf, k1, b):
    """Term-wise concentration factor 0..1 from BM25 TF saturation.

    For each term, compare the observed BM25 TF-saturation sum with the
    same sum evaluated at length-proportional fractional expected counts.
    BM25's term-IDF and (k1+1) factors cancel in this ratio.  The result
    can thus modify any local scorer's sum, including SimpleRidf, without
    changing that scorer's local formula.
    """
    lengths = corpus.doc_len[corpus.doc_ids].astype(np.float64)
    norm = k1 * (1.0 - b + b * lengths / corpus.avg_doc_len)
    all_norm = k1 * (1.0 - b + b * corpus.doc_len / corpus.avg_doc_len)
    observed = np.bincount(
        tid, weights=tf / (tf + all_norm[did]), minlength=len(corpus.cf)
    )
    null = np.zeros(len(corpus.cf), dtype=np.float64)
    proportions = lengths / corpus.collection_len
    # Batch over terms so no dense corpus-wide contingency matrix is needed.
    for start in range(1, len(null), 256):
        end = min(start + 256, len(null))
        expected = corpus.cf[start:end, None] * proportions[None, :]
        null[start:end] = np.sum(expected / (expected + norm[None, :]), axis=1)
    loss = np.zeros(len(null), dtype=np.float64)
    np.divide(observed, null, out=loss, where=null > 0)
    return np.clip(1.0 - loss, 0.0, 1.0)


def local_scores(corpus, code, term_ids, doc_ids, freqs, starts, ends, args):
    """Compute document-level scores using scorers.py, without aggregation."""
    lookup = code
    if code == "bm25":
        lookup = f"bm25k{args.k1:g}b{args.b:g}e{args.idf_exp:g}"
    elif code == "simpleridf" and args.simple_k is not None:
        lookup = f"simpleridfk{args.simple_k:g}"
    scorer = make_scorer(corpus, lookup)

    local = np.empty(len(freqs), dtype=np.float64)
    for start, end in zip(starts, ends):
        doc_id = int(doc_ids[start])
        local[start:end] = scorer.score_terms(doc_id, term_ids[start:end], freqs[start:end])

    if not np.all(np.isfinite(local)):
        raise ValueError(f"{code}: nonfinite local scores in scorers.py")

    if isinstance(scorer, BM25):
        # Restore the factor omitted by scorers.py, for Java BM25 equivalence.
        local *= scorer.k1 + 1.0
    return local, isinstance(scorer, BM25)


def aggregate(term_ids, freqs, local, vocab_size, method, power, loss=None):
    """Aggregate signed document scores using the requested method."""
    if method == "sum":
        return np.bincount(term_ids, weights=local, minlength=vocab_size)
    if method == "tf-l2":
        weights = freqs * np.maximum(local, 0.0) ** 2
        return np.sqrt(np.bincount(term_ids, weights=weights, minlength=vocab_size))
    if method == "power":
        # Signed extension of the power sum for contrastive scorers.
        total = np.bincount(
            term_ids,
            weights=np.sign(local) * np.abs(local) ** power,
            minlength=vocab_size,
        )
        return np.sign(total) * np.abs(total) ** (1.0 / power)
    if method == "satloss":
        if loss is None:
            raise ValueError("satloss requires a dispersion-loss vector")
        return np.bincount(term_ids, weights=local, minlength=vocab_size) * loss
    raise ValueError(f"Unknown aggregator: {method}")


def write_results(path, ranking, lemmas, cf, df, scores):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, delimiter="\t")
        writer.writerow(("rank", "term_id", "lemma", "score", "cf", "df"))
        for rank, term_id in enumerate(ranking, 1):
            writer.writerow((
                rank, int(term_id), lemmas[term_id], f"{scores[term_id]:.9f}",
                int(cf[term_id]), int(df[term_id])
            ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="TSV file prefix, e.g. ./ddr-")
    parser.add_argument("--scorers", nargs="+", default=["kl", "bm25", "simpleridf"],
                        help="kl, js, jslog, or any scorer code supported by scorers.py")
    parser.add_argument("--js-tau", type=float, default=500.0,
                        help="positive CF transition scale for jslog, default 500")
    parser.add_argument("--families", action="store_true",
                        help="print one inferred term neighborhood per ranked keyword")
    parser.add_argument("--family-size", type=int, default=5,
                        help="maximum terms per neighborhood, including its seed (default 5)")
    parser.add_argument("--family-sim", type=float, default=0.30,
                        help="minimum length-adjusted cosine similarity (default 0.30)")
    parser.add_argument("--family-seeds", nargs="+", default=None,
                        help="inspect specified lemma neighborhoods instead of ranked seeds")
    parser.add_argument("--aggregators", nargs="+", choices=["sum", "tf-l2", "power", "satloss"],
                        default=None, help="apply each selected aggregator to each local scorer; "
                        "omit to preserve previous scorer-specific defaults")
    parser.add_argument("--power", type=float, default=1.2,
                        help="power-sum exponent, default 1.2")
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--idf-exp", type=float, default=0.9)
    parser.add_argument("--simple-k", type=float, default=None)
    parser.add_argument("--vocab", nargs="+", choices=VOCABS, default=["all"],
                        help="vocabulary filters; default: all (same as keywords-glob.py)")
    parser.add_argument("--stopwords", type=Path, default=DEFAULT_STOPWORDS,
                        help="UTF-8 stopword list for stopword-dependent vocabularies; "
                        "default: stopwords.txt beside script")
    args = parser.parse_args()

    if not np.isfinite(args.js_tau) or args.js_tau <= 0:
        parser.error("--js-tau must be positive and finite")
    if args.family_size < 2 or not 0 <= args.family_sim <= 1:
        parser.error("--family-size must be >= 2 and --family-sim must be in [0,1]")
    if args.family_seeds and not args.families:
        parser.error("--family-seeds requires --families")
    if not np.isfinite(args.power) or args.power < 1.0:
        parser.error("--power must be finite and >= 1")
    if args.top <= 0 or not np.isfinite(args.k1) or args.k1 <= 0:
        parser.error("--top and --k1 must be positive")
    if not 0 <= args.b <= 1 or not np.isfinite(args.idf_exp) or args.idf_exp < 0:
        parser.error("--b must be in [0,1] and --idf-exp must be nonnegative")
    if args.simple_k is not None and (not np.isfinite(args.simple_k) or args.simple_k <= 0):
        parser.error("--simple-k must be a positive finite number")

    vocabularies = list(dict.fromkeys(args.vocab))
    stopwords = set()
    if any(mode in STOPWORD_VOCABS for mode in vocabularies):
        if not args.stopwords.is_file():
            parser.error(f"Stopword file not found: {args.stopwords} "
                         "(use --stopwords to select one)")
        stopwords = load_stopwords(args.stopwords)

    corpus, ids, lemmas, tid, did, tf = load(args.prefix)
    args.out.mkdir(parents=True, exist_ok=True)
    masks = {mode: vocabulary_mask(lemmas, stopwords, mode) for mode in vocabularies}
    print(f"Corpus: {corpus.n_docs} docs, {len(ids)} terms, "
          f"{corpus.collection_len} tokens, {len(tf)} postings", flush=True)

    order = np.argsort(did, kind="stable")
    doc_ids, term_ids, freqs = did[order], tid[order], tf[order]
    starts = np.flatnonzero(np.r_[True, doc_ids[1:] != doc_ids[:-1]])
    ends = np.r_[starts[1:], len(freqs)]

    loss = None
    if args.aggregators is not None and "satloss" in args.aggregators:
        loss = bm25_saturation_loss(corpus, tid, did, tf, args.k1, args.b)

    family_profiles = None
    for name in dict.fromkeys(args.scorers):
        if name in ("kl", "js", "jslog"):
            # Corpus-level divergences do not use document-score aggregators.
            scores = (kl(corpus, tid, did, tf) if name == "kl"
                      else js(corpus, tid, did, tf) if name == "js"
                      else jslog(corpus, tid, did, tf, args.js_tau))
            label = f"jslog-tau{args.js_tau:g}" if name == "jslog" else name
            rankings = [(label, scores)]
        else:
            try:
                local, is_bm25 = local_scores(
                    corpus, name, term_ids, doc_ids, freqs, starts, ends, args
                )
            except ValueError as exc:
                parser.error(str(exc))
            methods = (dict.fromkeys(args.aggregators) if args.aggregators is not None
                       else ["sum" if is_bm25 else "tf-l2"])
            rankings = [(
                f"{name}-{method}{args.power:g}" if args.aggregators is not None
                and method == "power" else
                (f"{name}-{method}" if args.aggregators is not None else name),
                aggregate(term_ids, freqs, local, len(corpus.cf), method,
                          args.power, loss)
            ) for method in methods]

        for label, scores in rankings:
            ranking = ids[np.argsort(-scores[ids], kind="stable")]
            for mode in vocabularies:
                filtered = ranking[masks[mode][ranking]]
                selected = filtered[:args.top]
                # Keep legacy filenames exactly for the unfiltered vocabulary.
                stem = f"corpus-{label}" + ("" if mode == "all" else f"-{mode}")
                display = label if mode == "all" else f"{label} / {mode}"
                if args.families:
                    if args.family_seeds:
                        by_name = {lemma: i for i, lemma in enumerate(lemmas)}
                        missing = [word for word in args.family_seeds if word not in by_name]
                        if missing:
                            parser.error(f"Unknown family seeds: {', '.join(missing)}")
                        selected = np.array([by_name[word] for word in args.family_seeds])
                    if family_profiles is None:
                        family_profiles = make_family_profiles(corpus, tid, did, tf)
                    families = term_neighborhoods(
                        corpus, family_profiles, selected, masks[mode],
                        args.family_size, args.family_sim
                    )
                    path = args.out / f"{stem}-families-top{len(selected)}.txt"
                    write_families(path, families, lemmas)
                    write_family_edges(path.with_suffix(".tsv"), families, lemmas)
                    print(f"\n{display} neighborhoods ({len(families)}):", flush=True)
                    for seed, neighbors in families:
                        words = ", ".join(lemmas[i] for i, _ in neighbors)
                        print(f"{lemmas[seed]}: {words}" if words else lemmas[seed], flush=True)
                    print(f"Saved: {path}", flush=True)
                else:
                    write_results(args.out / f"{stem}-all.tsv", filtered, lemmas,
                                  corpus.cf, corpus.df, scores)
                    write_results(args.out / f"{stem}-top{args.top}.tsv", selected,
                                  lemmas, corpus.cf, corpus.df, scores)
                    print(f"\n{display} top {len(selected)}:", flush=True)
                    print(", ".join(lemmas[term_id] for term_id in selected), flush=True)


if __name__ == "__main__":
    main()
