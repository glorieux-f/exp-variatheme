#!/usr/bin/env python3
"""Rank keywords in a glob-selected document set using existing scorers.

The three aggregation methods share the original corpus statistics:

* pool: score the sum of selected term counts as a virtual document.
* mean: average scores over selected documents (absent terms contribute 0).
* centroid: L2-normalize each sparse score vector, then average.

Requires scorers.py in the same directory and its NumPy/SciPy dependencies.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from scorers import make_scorer


DEFAULT_STOPWORDS = Path(__file__).with_name("stopwords.txt")
DEFAULT_SCORERS = ("tfidf", "subtfidfa0.3", "poissonidfa1", "txm", "poissondev")
VOCABS = ("all", "content", "nostops", "nocaps", "stops", "caps", "stopscaps")
AGGREGATIONS = ("pool", "mean", "centroid")

def read_rows(path: Path, required: set[str]):
    """Yield TSV records with a checked header and physical row number."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            yield line, row


def load_documents(path: Path, pattern: str):
    """Load document lengths and select IDs using case-sensitive glob matching."""
    lengths = {}
    selected = []
    for line, row in read_rows(path, {"doc_id", "identifier", "doc_len"}):
        doc_id = int(row["doc_id"])
        length = int(row["doc_len"])
        if doc_id < 1 or length < 0 or doc_id in lengths:
            raise ValueError(f"{path}:{line}: invalid or duplicate doc_id/doc_len")
        lengths[doc_id] = length
        if fnmatch.fnmatchcase(row["identifier"], pattern):
            selected.append(doc_id)
    if not selected:
        raise ValueError(f"No document identifiers match {pattern!r}")
    return lengths, selected


def load_terms(path: Path):
    """Load corpus-wide frequencies and lemmas indexed by term ID."""
    terms = {}
    for line, row in read_rows(path, {"term_id", "lemma", "cf", "df"}):
        term_id = int(row["term_id"])
        cf, df = int(row["cf"]), int(row["df"])
        if term_id < 1 or cf < 0 or df < 0 or term_id in terms:
            raise ValueError(f"{path}:{line}: invalid or duplicate term statistics")
        terms[term_id] = (row["lemma"], cf, df)
    return terms


def load_postings(path: Path, selected: list[int], lengths: dict[int, int], terms: dict):
    """Read only the selected postings, retaining sparse per-document counts."""
    counts = {doc_id: Counter() for doc_id in selected}
    pooled = Counter()
    for line, row in read_rows(path, {"term_id", "doc_id", "tf"}):
        doc_id = int(row["doc_id"])
        if doc_id not in counts:
            continue
        term_id, tf = int(row["term_id"]), int(row["tf"])
        if term_id not in terms or tf <= 0:
            raise ValueError(f"{path}:{line}: invalid term_id/tf")
        counts[doc_id][term_id] += tf
        pooled[term_id] += tf
    for doc_id, frequencies in counts.items():
        actual = sum(frequencies.values())
        if actual != lengths[doc_id]:
            raise ValueError(
                f"{path}: document {doc_id}: postings sum {actual} != doc_len {lengths[doc_id]}"
            )
    return counts, pooled


class Corpus:
    """Original global statistics with one appended, non-member virtual document."""

    def __init__(self, lengths: dict[int, int], terms: dict[int, tuple[str, int, int]],
                 virtual_length: int, postings: dict[int, Counter[int]], pooled: Counter[int]):
        self.n_docs = len(lengths)
        self.collection_len = sum(lengths.values())
        self.avg_doc_len = self.collection_len / self.n_docs
        self.virtual_id = max(lengths) + 1
        self.doc_len = np.zeros(self.virtual_id + 1, dtype=np.int64)
        for doc_id, length in lengths.items():
            self.doc_len[doc_id] = length
        self.doc_len[self.virtual_id] = virtual_length
        self.cf = np.zeros(max(terms) + 1, dtype=np.int64)
        self.df = np.zeros(max(terms) + 1, dtype=np.int64)
        for term_id, (_, cf, df) in terms.items():
            self.cf[term_id] = cf
            self.df[term_id] = df
        self.postings = postings
        self.pooled = pooled

    def tf(self, term_id: int, doc_id: int) -> int:
        """Return term frequency for an original or virtual document."""
        if doc_id == self.virtual_id:
            return self.pooled.get(term_id, 0)
        return self.postings.get(doc_id, {}).get(term_id, 0)


def normalize_word(word: str) -> str:
    """Return a normalized form for stopword detection."""
    return unicodedata.normalize("NFC", word.strip()).casefold()


def load_stopwords(path: Path) -> set[str]:
    """Load one stopword per line, ignoring comments and headings."""
    words = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            word = line.strip()
            if word and not word.startswith("#") and word != "__STOPWORDS":
                words.add(normalize_word(word))
    return words


def vocabulary_mask(lemmas: list[str], stopwords: set[str], mode: str) -> np.ndarray:
    """Select vocabulary without changing the scorer's statistics or normalization."""
    if mode == "all":
        return np.ones(len(lemmas), dtype=bool)
    caps = np.fromiter((bool(s[:1].isupper()) for s in lemmas), dtype=bool)
    stops = np.fromiter((normalize_word(s) in stopwords for s in lemmas), dtype=bool)
    return {
        "content": ~caps & ~stops,
        "nostops": ~stops,
        "nocaps": ~caps,
        "stops": stops,
        "caps": caps,
        "stopscaps": stops | caps,
    }[mode]


def scorer_results(corpus: Corpus, postings: dict[int, Counter[int]],
                   pooled: Counter[int], scorers: list[str], aggregations: list[str],
                   terms: dict[int, tuple[str, int, int]]):
    """Score candidate terms, returning full sorted rankings for each mode."""
    candidate_ids = np.asarray(sorted(pooled), dtype=np.int64)
    candidate_tf = np.asarray([pooled[int(t)] for t in candidate_ids], dtype=np.int64)
    names = [terms[int(t)][0] for t in candidate_ids]
    positions = {int(term_id): i for i, term_id in enumerate(candidate_ids)}
    documents = []
    if "mean" in aggregations or "centroid" in aggregations:
        for doc_id, counts in postings.items():
            ids = np.asarray(sorted(counts), dtype=np.int64)
            tf = np.asarray([counts[int(t)] for t in ids], dtype=np.int64)
            indices = np.asarray([positions[int(t)] for t in ids], dtype=np.int64)
            documents.append((doc_id, ids, tf, indices))

    results = {}
    for code in scorers:
        scorer = make_scorer(corpus, code)
        if ("pool" in aggregations and corpus.doc_len[corpus.virtual_id] == corpus.collection_len
                and scorer.requires_complement):
            raise ValueError(f"Pooled corpus equals the entire corpus: {code} "
                             "requires a non-empty comparison corpus")
        sums = {}
        if "pool" in aggregations:
            sums["pool"] = scorer.score_terms(corpus.virtual_id, candidate_ids, candidate_tf)
        if "mean" in aggregations or "centroid" in aggregations:
            mean = np.zeros(len(candidate_ids), dtype=np.float64)
            centroid = np.zeros(len(candidate_ids), dtype=np.float64)
            for doc_id, ids, tf, indices in documents:
                values = scorer.score_terms(doc_id, ids, tf)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Non-finite {code} score for document ID {doc_id}")
                if "mean" in aggregations:
                    mean[indices] += values
                if "centroid" in aggregations:
                    norm = float(np.linalg.norm(values))
                    if norm > 0.0:
                        centroid[indices] += values / norm
            if "mean" in aggregations:
                sums["mean"] = mean / len(postings)
            if "centroid" in aggregations:
                sums["centroid"] = centroid / len(postings)
        for aggregation, values in sums.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite {code} score for {aggregation}")
            order = sorted(
                range(len(candidate_ids)),
                key=lambda i: (-float(values[i]), names[i].casefold(), names[i], int(candidate_ids[i])),
            )
            results[(code, aggregation)] = order
    return names, results


def pattern_slug(pattern: str) -> str:
    """Encode a document glob as a Windows-safe filename component."""
    result = []
    for c in pattern:
        if c == "*":
            result.append("~")
        elif c == "?":
            result.append("~q")
        elif c in '<>:"/\\|\x00':
            result.append(f"~{ord(c):02x}")
        elif ord(c) < 32:
            result.append(f"~{ord(c):02x}")
        else:
            result.append(c)
    return "".join(result).rstrip(". ") or "selection"


def build(table_prefix: Path, pattern: str, output_dir: Path, scorers: list[str],
          vocabularies: list[str], aggregations: list[str], stopwords_path: Path,
          top: int) -> Path:
    """Generate all requested keyword lists in a single two-line-block file."""
    prefix = table_prefix.name
    if not prefix:
        raise ValueError("table_prefix must include a filename prefix (e.g. ddr-)")
    directory = table_prefix.parent
    lengths, selected = load_documents(directory / f"{prefix}docs.tsv", pattern)
    terms = load_terms(directory / f"{prefix}terms.tsv")
    postings, pooled = load_postings(
        directory / f"{prefix}postings.tsv", selected, lengths, terms
    )
    stopwords = load_stopwords(stopwords_path)
    virtual_length = sum(lengths[doc_id] for doc_id in selected)
    corpus = Corpus(lengths, terms, virtual_length, postings, pooled)
    if not pooled:
        raise ValueError(f"Selected documents matching {pattern!r} contain no term postings")

    names, rankings = scorer_results(corpus, postings, pooled, scorers, aggregations, terms)
    masks = {mode: vocabulary_mask(names, stopwords, mode) for mode in vocabularies}
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    stem = f"{prefix}{pattern_slug(pattern)}-{timestamp}"
    output = output_dir / f"{stem}.txt"
    suffix = 2
    while output.exists():
        output = output_dir / f"{stem}-{suffix}.txt"
        suffix += 1
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        first = True
        for mode in VOCABS:
            if mode not in masks:
                continue
            mask = masks[mode]
            for code in scorers:
                for aggregation in AGGREGATIONS:
                    if aggregation not in aggregations:
                        continue
                    ranking = rankings[(code, aggregation)]
                    ranked = (names[i] for i in ranking if mask[i])
                    keywords = list(ranked) if top == 0 else list(_first(ranked, top))
                    if not first:
                        stream.write("\n")
                    first = False
                    stream.write(f"{pattern} – {mode} – {code} – {aggregation}\n")
                    stream.write(", ".join(keywords) + "\n")
    print(f"Selected {len(selected)} document(s) matching {pattern!r}")
    print(output)
    return output


def _first(items, count: int):
    """Yield at most count items from a ranked sequence."""
    for i, item in enumerate(items):
        if i == count:
            break
        yield item


def main() -> None:
    """Parse command-line arguments and write the keyword file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_prefix", type=Path, help="Contingency filename prefix, e.g. data/ddr-")
    parser.add_argument("identifier_glob", help="Case-sensitive document identifier glob")
    parser.add_argument("output_dir", type=Path, help="Directory for the output text file")
    parser.add_argument("--scorer", nargs="+", default=list(DEFAULT_SCORERS),
                        help="Scorers registered in scorers.py (default: five families; G² variants are opt-in)")
    parser.add_argument("--vocab", nargs="+", choices=VOCABS, default=["all"],
                        help="Vocabulary filters (default: all)")
    parser.add_argument("--aggreg", nargs="+", choices=AGGREGATIONS,
                        default=list(AGGREGATIONS), help="Aggregation modes (default: all three)")
    parser.add_argument("--stopwords", type=Path, default=DEFAULT_STOPWORDS,
                        help="Stopword file (default: stopwords.txt beside script)")
    parser.add_argument("--top", type=int, default=100,
                        help="Keywords per list; 0 means all (default: 100)")
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be >= 0")
    args.scorer = list(dict.fromkeys(code.strip().lower() for code in args.scorer))
    try:
        build(args.table_prefix, args.identifier_glob, args.output_dir, args.scorer,
              list(dict.fromkeys(args.vocab)), list(dict.fromkeys(args.aggreg)),
              args.stopwords, args.top)
    except (OSError, ValueError, ZeroDivisionError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
