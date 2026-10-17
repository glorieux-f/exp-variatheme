#!/usr/bin/env python3
"""Summarize corpus rarity and stopword contamination of ranked keyword lists.

Input keyword files use the output format of keywords.py:

    [document_id] metadata
    keyword1, keyword2, ...

The delimiter is comma + space, NOT a bare comma: some lemmas are numeric
strings containing commas (e.g. 0,6).

Each occurrence of a keyword in one document's list counts once. Statistics
are pooled across documents for each scorer, possibly across author files.
Only listed keywords contribute; there is no automatic stopword filtering.

The summary is written to stdout (and optionally -o). A separate stopword
report lists the most commonly selected stopwords per scorer, sorted by the
number of documents whose top-N list contains each stopword. With -o, this
report is saved as <output stem>-stopwords.tsv; otherwise it is displayed
on stderr or saved to --stopwords-output if specified.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import glob
from pathlib import Path
import re
from statistics import median
import sys
import unicodedata
from collections.abc import Iterator


FILENAME_RE = re.compile(
    r"^.+?-keywords\d+-(?:all|content|nostops|nocaps|stops|caps|stopscaps)"
    r"(?:-min\d+)?-(?P<scorer>.+)\.txt$",
    re.IGNORECASE,
)
DOC_RE = re.compile(r"^\[([^]]+)\]")


def normalized(word: str) -> str:
    return unicodedata.normalize("NFC", word.strip())


def load_terms(path_or_prefix: Path) -> dict[str, tuple[int, int]]:
    """Load lemma -> (CF, DF) from terms.tsv or {prefix}terms.tsv."""
    path = path_or_prefix
    if not path.is_file():
        path = Path(f"{path_or_prefix}terms.tsv")
    if not path.is_file():
        raise FileNotFoundError(f"Term statistics not found: {path_or_prefix}")

    terms: dict[str, tuple[int, int]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        fields = stream.readline().rstrip("\r\n").split("\t")
        try:
            i_lemma, i_cf, i_df = (fields.index(x) for x in ("lemma", "cf", "df"))
        except ValueError as exc:
            raise ValueError(f"{path}: expected TSV columns lemma, cf, df") from exc
        for lineno, line in enumerate(stream, 2):
            if not line.strip():
                continue
            cells = line.rstrip("\r\n").split("\t")
            try:
                lemma = normalized(cells[i_lemma])
                value = (int(cells[i_cf]), int(cells[i_df]))
            except (IndexError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid term row") from exc
            if lemma in terms:
                raise ValueError(f"{path}:{lineno}: duplicate lemma {lemma!r}")
            terms[lemma] = value
    return terms


def load_stopwords(path: Path) -> set[str]:
    """Ignore comments and the __STOPWORDS section header; match casefolded."""
    words: set[str] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        word = normalized(line)
        if not word or word.startswith("#") or word == "__STOPWORDS":
            continue
        words.add(word.casefold())
    return words


def paths_from_patterns(patterns: list[str]) -> list[Path]:
    """Expand quoted glob patterns on Windows or POSIX; accept directories."""
    result: dict[Path, None] = {}
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_dir():
            matches = candidate.glob("*-keywords*.txt")
        else:
            matches = (Path(p) for p in glob.glob(pattern, recursive=True))
        for match in matches:
            if match.is_file():
                result[match.resolve()] = None
    if not result:
        raise ValueError("No keyword files matched the supplied paths/globs")
    return sorted(result)


def read_keyword_lists(path: Path) -> Iterator[tuple[str, list[str]]]:
    """Stream the keywords.py document blocks and deduplicate within lists."""
    document_id: str | None = None
    with path.open(encoding="utf-8-sig") as stream:
        for lineno, line in enumerate(stream, 1):
            data = line.strip()
            if document_id is None:
                if not data:
                    continue
                match = DOC_RE.match(data)
                if match is None:
                    raise ValueError(f"{path}:{lineno}: expected [document_id] metadata")
                document_id = match.group(1)
            else:
                if data.startswith("["):
                    raise ValueError(f"{path}:{lineno}: missing keyword line for {document_id}")
                # The generator uses ', ' as delimiter; do not split decimal commas.
                keywords = [normalized(w) for w in data.split(", ") if w.strip()]
                yield document_id, list(dict.fromkeys(keywords))
                document_id = None
    if document_id is not None:
        raise ValueError(f"{path}: missing keyword line for {document_id}")


@dataclass
class Summary:
    documents: int = 0
    files: int = 0
    keyword_count: int = 0
    known_count: int = 0
    cf1_count: int = 0
    dfs: list[int] = field(default_factory=list)
    stop_count: int = 0
    early_count: int = 0
    early_stop_count: int = 0
    docs_with_early_stop: int = 0
    missing: dict[str, int] = field(default_factory=dict)
    seen_documents: set[str] = field(default_factory=set)
    # Aggregate case-insensitively: e.g. "le" and "Le" are one stopword.
    stop_occurrences: Counter[str] = field(default_factory=Counter)
    stop_documents: Counter[str] = field(default_factory=Counter)
    stop_early_occurrences: Counter[str] = field(default_factory=Counter)
    stop_rank_sums: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        doc_id: str,
        keywords: list[str],
        terms: dict[str, tuple[int, int]],
        stops: set[str],
        top: int,
        early: int,
        filename: Path,
    ) -> None:
        if doc_id in self.seen_documents:
            raise ValueError(f"{filename}: repeated document {doc_id!r} for this scorer")
        self.seen_documents.add(doc_id)
        self.documents += 1
        selected = keywords[:top]
        self.keyword_count += len(selected)
        early_selected = selected[:early]
        self.early_count += len(early_selected)
        stop_flags = [word.casefold() in stops for word in selected]
        self.stop_count += sum(stop_flags)
        n_early_stops = sum(stop_flags[:early])
        self.early_stop_count += n_early_stops
        self.docs_with_early_stop += n_early_stops > 0

        seen_stops_in_doc: set[str] = set()
        for rank, word in enumerate(selected, 1):
            folded = word.casefold()
            if folded in stops:
                self.stop_occurrences[folded] += 1
                self.stop_rank_sums[folded] += rank
                if rank <= early:
                    self.stop_early_occurrences[folded] += 1
                seen_stops_in_doc.add(folded)
            # A word may have differently capitalized forms in one document;
            # document prevalence must still count that document only once.
            stats = terms.get(word)
            if stats is None:
                self.missing[word] = self.missing.get(word, 0) + 1
                continue
            cf, df = stats
            self.known_count += 1
            self.cf1_count += cf == 1
            self.dfs.append(df)
        self.stop_documents.update(seen_stops_in_doc)


def pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.2f}" if denominator else "NA"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keywords", nargs="+", help="Keyword .txt files, directories or quoted globs (multiple allowed)")
    parser.add_argument("--terms", required=True, type=Path, help="terms.tsv or a corpus prefix such as ../data/3_contingeny/ddr-")
    parser.add_argument("--stopwords", required=True, type=Path, help="Stopword file, one entry per line")
    parser.add_argument("--top", type=int, default=100, help="Keywords considered per document (default: 100)")
    parser.add_argument("--early", type=int, default=10, help="Top ranks used for early-stopword metrics (default: 10)")
    parser.add_argument("-o", "--output", type=Path, help="Write summary TSV here as well as stdout")
    parser.add_argument(
        "--stopwords-top", type=int, default=20,
        help="Number of most frequently selected stopwords per scorer (default: 20)",
    )
    parser.add_argument(
        "--stopwords-output", type=Path,
        help="Detailed stopword TSV (default: <output stem>-stopwords.tsv when -o is used)",
    )
    args = parser.parse_args()
    if args.top < 1 or args.early < 1 or args.early > args.top:
        parser.error("Require --top >= --early >= 1")
    if args.stopwords_top < 1:
        parser.error("Require --stopwords-top >= 1")

    term_stats = load_terms(args.terms)
    stopwords = load_stopwords(args.stopwords)
    files = paths_from_patterns(args.keywords)
    by_scorer: dict[str, Summary] = {}

    for path in files:
        match = FILENAME_RE.match(path.name)
        if not match:
            raise ValueError(f"Unrecognized keyword file name: {path.name}")
        scorer = match.group("scorer")
        acc = by_scorer.setdefault(scorer, Summary())
        acc.files += 1
        for doc_id, words in read_keyword_lists(path):
            acc.add(doc_id, words, term_stats, stopwords, args.top, args.early, path)

    headers = (
        "scorer", "files", "docs", f"keywords_top{args.top}",
        f"cf1_pct_top{args.top}", f"median_df_top{args.top}",
        f"stop_pct_top{args.top}", f"stop_pct_first{args.early}",
        f"docs_with_stop_first{args.early}_pct", "unknown_keywords",
    )
    rows = ["\t".join(headers)]
    for scorer, acc in sorted(by_scorer.items()):
        med = median(acc.dfs) if acc.dfs else None
        row = (
            scorer, str(acc.files), str(acc.documents), str(acc.keyword_count),
            pct(acc.cf1_count, acc.known_count),
            f"{med:g}" if med is not None else "NA",
            pct(acc.stop_count, acc.keyword_count),
            pct(acc.early_stop_count, acc.early_count),
            pct(acc.docs_with_early_stop, acc.documents),
            str(sum(acc.missing.values())),
        )
        rows.append("\t".join(row))
        if acc.missing:
            examples = ", ".join(f"{w} ({n})" for w, n in sorted(acc.missing.items(), key=lambda kv: -kv[1])[:6])
            print(f"Warning: {scorer}: {sum(acc.missing.values())} keyword occurrences not found in terms.tsv: {examples}", file=sys.stderr)

    report = "\n".join(rows) + "\n"
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8", newline="\n")
        print(f"Written: {args.output}", file=sys.stderr)

    # Preserve the original summary TSV layout. Produce a second, long-form
    # TSV with the most common stopwords selected by each scorer.
    details = ["\t".join((
        "scorer", "rank", "stopword", f"documents_top{args.top}",
        f"docs_pct_top{args.top}", f"occurrences_top{args.top}",
        f"occurrences_first{args.early}", f"mean_rank_top{args.top}",
    ))]
    for scorer, acc in sorted(by_scorer.items()):
        ordered = sorted(
            acc.stop_occurrences,
            key=lambda w: (-acc.stop_documents[w], -acc.stop_occurrences[w], w),
        )[:args.stopwords_top]
        # Rank by number of documents containing a stopword in the selected
        # keyword list, not its total corpus frequency.
        for rank, word in enumerate(ordered, 1):
            occurrences = acc.stop_occurrences[word]
            details.append("\t".join((
                scorer, str(rank), word,
                str(acc.stop_documents[word]),
                pct(acc.stop_documents[word], acc.documents),
                str(occurrences), str(acc.stop_early_occurrences[word]),
                f"{acc.stop_rank_sums[word] / occurrences:.1f}",
            )))
        overview = ", ".join(
            f"{word} ({acc.stop_documents[word]})" for word in ordered
        ) or "(none)"
        print(f"{scorer}, top {args.stopwords_top} stopwords (documents): {overview}", file=sys.stderr)

    stopwords_report = "\n".join(details) + "\n"
    details_path = args.stopwords_output
    if details_path is None and args.output:
        details_path = args.output.with_name(args.output.stem + "-stopwords.tsv")
    if details_path:
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(stopwords_report, encoding="utf-8", newline="\n")
        print(f"Written: {details_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
