#!/usr/bin/env python3
"""Build frequency lists from a term-document contingency table.

Four vocabulary variants are written:

    all       keep capitalized lemmas and stopwords
    nocaps    exclude capitalized lemmas, keep stopwords
    nostops   keep capitalized lemmas, exclude stopwords
    content   exclude capitalized lemmas and stopwords

By default, aggregate all documents. Use --select to restrict the aggregation
by document identifier, with shell-style patterns such as ddr19721100ao*.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import re
import unicodedata
from collections import Counter
from pathlib import Path


DEFAULT_STOPWORDS = Path(__file__).with_name("stopwords.txt")
VOCABS = ("all", "nocaps", "nostops", "content")


def normalize_word(value: str) -> str:
    """Return a case-insensitive normalized form for stopword matching."""
    return unicodedata.normalize("NFC", value.strip()).casefold()


def pattern_slug(value: str) -> str:
    """Make a glob pattern safe for filenames, including on Windows."""
    value = value.replace("*", "~").replace("?", "_")
    return re.sub(r"[^a-zA-Z0-9._~\[\]-]+", "-", value).strip("-.")


def load_stopwords(path: Path) -> set[str]:
    """Load a one-entry-per-line stopword list."""
    words: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            value = line.strip()
            if not value or value.startswith("#") or value == "__STOPWORDS":
                continue
            words.add(normalize_word(value))
    return words


def load_terms(path: Path) -> dict[int, str]:
    """Load term_id -> lemma from terms.tsv."""
    terms: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"term_id", "lemma"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            terms[int(row["term_id"])] = row["lemma"]
    return terms


def load_documents(path: Path, select: str | None) -> set[int]:
    """Load IDs of documents whose identifiers match the selected pattern."""
    doc_ids: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"doc_id"} | ({"identifier"} if select is not None else set())
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if select is None or fnmatch.fnmatchcase(row["identifier"], select):
                doc_ids.add(int(row["doc_id"]))
    if not doc_ids:
        raise ValueError(f"{path}: no documents match {select!r}")
    return doc_ids


def keep_lemma(lemma: str, vocab: str, stopwords: set[str]) -> bool:
    """Return whether a lemma belongs to one vocabulary variant."""
    is_cap = bool(lemma[:1].isupper())
    is_stop = normalize_word(lemma) in stopwords

    if vocab == "all":
        return True
    if vocab == "nocaps":
        return not is_cap
    if vocab == "nostops":
        return not is_stop
    if vocab == "content":
        return not is_cap and not is_stop
    raise ValueError(f"Unknown vocabulary: {vocab}")


def aggregate(
    path: Path, doc_ids: set[int], terms: dict[int, str], select: str | None
) -> tuple[Counter[int], Counter[int]]:
    """Aggregate term and document frequencies over selected documents."""
    cf: Counter[int] = Counter()
    df: Counter[int] = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"term_id", "doc_id", "tf"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")

        for line_no, row in enumerate(reader, 2):
            doc_id = int(row["doc_id"])
            if doc_id not in doc_ids:
                if select is None:
                    raise ValueError(f"{path}:{line_no}: unknown doc_id {doc_id}")
                continue

            term_id = int(row["term_id"])
            tf = int(row["tf"])
            if term_id not in terms:
                raise ValueError(f"{path}:{line_no}: unknown term_id {term_id}")
            if tf <= 0:
                raise ValueError(f"{path}:{line_no}: tf must be > 0")

            cf[term_id] += tf
            df[term_id] += 1

    return cf, df


def write_frequency_list(
    path: Path,
    vocab: str,
    n_docs: int,
    terms: dict[int, str],
    cf: Counter[int],
    df: Counter[int],
    stopwords: set[str],
    top: int,
) -> None:
    """Write terms in descending frequency, normalized by all selected tokens."""
    rows = [
        (terms[term_id], frequency, df[term_id])
        for term_id, frequency in cf.items()
        if keep_lemma(terms[term_id], vocab, stopwords)
    ]
    rows.sort(key=lambda row: (-row[1], row[0].casefold(), row[0]))

    total_cf = sum(cf.values())
    if top > 0:
        rows = rows[:top]

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        if n_docs == 1:
            writer.writerow(("rank", "lemma", "cf", "cf_pm"))
        else:
            writer.writerow(("rank", "lemma", "cf", "df", "mean_tf", "cf_pm", "df_pct"))

        for rank, (lemma, frequency, document_frequency) in enumerate(rows, 1):
            cf_pm = 1_000_000 * frequency / total_cf if total_cf else 0.0
            if n_docs == 1:
                writer.writerow((rank, lemma, frequency, f"{cf_pm:.6f}"))
            else:
                mean_tf = frequency / document_frequency
                df_pct = 100 * document_frequency / n_docs
                writer.writerow(
                    (rank, lemma, frequency, document_frequency,
                     f"{mean_tf:.6f}", f"{cf_pm:.6f}", f"{df_pct:.6f}")
                )


def build(input_prefix: Path, output_dir: Path, stopwords_path: Path, top: int, select: str | None) -> None:
    """Write frequency lists for the corpus or the selected identifiers."""
    if input_prefix.is_dir():
        directory, prefix = input_prefix, ""
    else:
        directory, prefix = input_prefix.parent, input_prefix.name

    terms = load_terms(directory / f"{prefix}terms.tsv")
    doc_ids = load_documents(directory / f"{prefix}docs.tsv", select)
    stopwords = load_stopwords(stopwords_path)
    cf, df = aggregate(directory / f"{prefix}postings.tsv", doc_ids, terms, select)

    output_dir.mkdir(parents=True, exist_ok=True)
    if select is not None:
        print(f"Selected {len(doc_ids)} document(s) matching {select!r}")

    name = pattern_slug(select) if select is not None else "corpus"
    for vocab in VOCABS:
        path = output_dir / f"{prefix}{name}-{vocab}.tsv"
        write_frequency_list(path, vocab, len(doc_ids), terms, cf, df, stopwords, top)
        print(path)


def main() -> None:
    """Parse command-line arguments and generate frequency lists."""
    parser = argparse.ArgumentParser(
        description="Build frequency lists for a corpus or selected document identifiers."
    )
    parser.add_argument(
        "input_prefix",
        type=Path,
        help="Contingency filename prefix (e.g. 3_contingency/ddr-) or unprefixed directory",
    )
    parser.add_argument("output_dir", type=Path, help="Directory for generated frequency lists")
    parser.add_argument(
        "--select",
        help="Document identifier or glob pattern (*, ?, [abc]); default: all documents",
    )
    parser.add_argument(
        "--stopwords",
        type=Path,
        default=DEFAULT_STOPWORDS,
        help=f"One-entry-per-line stopword list (default: {DEFAULT_STOPWORDS})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1000,
        help="Maximum terms per list; 0 writes the complete vocabulary (default: 1000)",
    )
    args = parser.parse_args()

    if args.top < 0:
        parser.error("--top must be >= 0")
    try:
        build(args.input_prefix, args.output_dir, args.stopwords, args.top, args.select)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
