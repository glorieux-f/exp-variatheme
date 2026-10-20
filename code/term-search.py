#!/usr/bin/env python3
r"""Interactive or one-shot term search over prefixed term-document TSV tables.

The input is a filename prefix: ``{prefix}docs.tsv``, ``{prefix}terms.tsv``
and ``{prefix}postings.tsv``. Scorers are resolved by ``scorers.make_scorer``.

Examples::

    python term-search.py guerre --input ..\data\3_contingency\ddr-
    python term-search.py guerre --input ..\data\3_contingency\ddr- --scorer bm25 g2 txm poissondev
    python term-search.py guerre paix --scorer tfidf subtfidfa0.3
    python term-search.py --scorer bm25 g2

With no lemma argument, the corpus is loaded once and an interactive prompt is
started. An empty line or EOF exits.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from corpus import Corpus
from scorers import Scorer, make_scorer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PREFIX = (SCRIPT_DIR / "../data/3_contingency/ddr-").resolve()
DEFAULT_SCORERS = ("tf",)


def format_number(value: float) -> str:
    """Format a scorer value compactly without hiding useful precision."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.8g}"


def print_result(corpus: Corpus, scorers: list[Scorer], lemma: str, limit: int | None) -> None:
    """Print corpus statistics and independently ranked documents for each scorer."""
    try:
        term_id = corpus.term_id(lemma)
    except KeyError:
        print(f"\n{lemma}\n  not found in vocabulary")
        return

    cf = int(corpus.cf[term_id])
    df = int(corpus.df[term_id])
    print(f"\n{lemma}")
    print(f"  occurrences: {cf}")
    print(f"  documents:   {df}")

    if df == 0:
        return

    doc_ids, tf = corpus.postings(term_id)
    if len(doc_ids) != df or int(tf.sum()) != cf:
        raise AssertionError(
            f"Inconsistent postings for {lemma!r}: "
            f"df={df}, postings={len(doc_ids)}, cf={cf}, sum(tf)={int(tf.sum())}"
        )

    for scorer in scorers:
        scores = np.asarray(scorer.score_docs(term_id, doc_ids, tf), dtype=np.float64)
        sortable = np.nan_to_num(scores, nan=-np.inf)

        # Primary: score descending. Tie: tf descending, then doc ID ascending.
        order = np.lexsort((doc_ids, -tf, -sortable))
        if limit is not None:
            order = order[:limit]

        print(f"\n  scorer:      {scorer.code} — {scorer.name}")
        print()
        print("rank\ttf\tscore\tidentifier\twork\ttitle")
        for rank, pos in enumerate(order, start=1):
            doc_id = int(doc_ids[pos])
            document = corpus.document(doc_id)
            print(
                f"{rank}\t{int(tf[pos])}\t{format_number(float(scores[pos]))}\t"
                f"{document.identifier}\t{document.work}\t{document.title}"
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Show corpus-local statistics and ranked documents for lemmas."
    )
    parser.add_argument(
        "lemmas",
        nargs="*",
        help="exact lemma(s); omit to enter interactive mode",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PREFIX,
        metavar="PREFIX",
        help=(
            "prefix for {prefix}docs.tsv, {prefix}terms.tsv and "
            f"{{prefix}}postings.tsv (default: {DEFAULT_INPUT_PREFIX})"
        ),
    )
    parser.add_argument(
        "--select",
        default="*",
        metavar="GLOB",
        help='document identifier glob defining the active corpus (default: "*")',
    )
    parser.add_argument(
        "--scorer",
        action="extend",
        nargs="+",
        default=[],
        metavar="CODE",
        help=(
            "one or more scorer codes registered in scorers.py; repeatable "
            f"(default: {' '.join(DEFAULT_SCORERS)})"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of documents to print per scorer (default: all)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def main() -> None:
    """Load the corpus once, select a subcorpus, and answer term queries."""
    args = parse_args()
    corpus = Corpus.load(args.input)
    if args.select != "*":
        corpus = corpus.select(args.select)

    # Fail early on invalid codes, and reuse scorer instances across lemmas.
    codes = args.scorer or DEFAULT_SCORERS
    scorers: list[Scorer] = []
    seen_codes: set[str] = set()
    for code in codes:
        scorer = make_scorer(corpus, code)
        if scorer.code not in seen_codes:
            scorers.append(scorer)
            seen_codes.add(scorer.code)

    print(
        f"Corpus: {args.select} — {corpus.n_docs:,} documents — "
        f"{corpus.collection_len:,} term occurrences\n"
        f"Document scorers: {', '.join(scorer.code for scorer in scorers)}"
    )

    if args.lemmas:
        for lemma in args.lemmas:
            print_result(corpus, scorers, lemma, args.limit)
        return

    while True:
        try:
            lemma = input("\nlemma> ").strip()
        except EOFError:
            print()
            return
        if not lemma:
            return
        print_result(corpus, scorers, lemma, args.limit)


if __name__ == "__main__":
    main()
