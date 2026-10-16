"""Rank likely text-reuse pairs among Markdown files.

The score is the fraction of the shorter document covered by shared six-word
sequences in sufficiently similar local windows. It ranges from 0 to 1.
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
SHINGLE = 6
WINDOW = 200
STEP = 100
ONES = b"\x01" * SHINGLE


@dataclass
class Window:
    doc: int
    start: int
    end: int
    shingles: set[tuple[str, ...]]


def read_words(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    text = re.sub(r"\A---[^\n]*\n.*?\n---[^\n]*\n", " ", text, flags=re.S)
    text = re.sub(r"(?m)^```[^\n]*\n.*?^```[^\n]*$", " ", text, flags=re.S | re.M)
    text = re.sub(r"(?<=\])\([^\n)]*\)", " ", text)
    return [match.group().casefold().replace("’", "'") for match in WORD.finditer(text)]


def make_windows(documents: list[list[str]]) -> list[Window]:
    windows = []
    for doc_id, words in enumerate(documents):
        for start in range(0, len(words), STEP):
            end = min(start + WINDOW, len(words))
            if end - start < 40:
                break
            shingles = {
                tuple(words[p:p + SHINGLE])
                for p in range(start, end - SHINGLE + 1)
            }
            windows.append(Window(doc_id, start, end, shingles))
    return windows


def find_scores(documents: list[list[str]], windows: list[Window],
                window_score: float, min_shared: int,
                max_frequency: int) -> dict[tuple[int, int], float]:
    inverted = defaultdict(list)
    for i, window in enumerate(windows):
        for shingle in window.shingles:
            inverted[shingle].append(i)

    coverage = {}
    for i, a in enumerate(windows):
        candidates = Counter()
        for shingle in a.shingles:
            posting = inverted[shingle]
            if len(posting) <= max_frequency:
                for j in posting:
                    if j > i and windows[j].doc != a.doc:
                        candidates[j] += 1

        for j, shared_count in candidates.items():
            b = windows[j]
            if (shared_count < min_shared or
                    shared_count / min(len(a.shingles), len(b.shingles)) < window_score):
                continue

            pair = (a.doc, b.doc)
            shorter = a if len(documents[a.doc]) <= len(documents[b.doc]) else b
            shared = a.shingles & b.shingles
            shared = {gram for gram in shared if len(inverted[gram]) <= max_frequency}
            mask = coverage.setdefault(pair, bytearray(len(documents[shorter.doc])))
            words = documents[shorter.doc]
            for pos in range(shorter.start, shorter.end - SHINGLE + 1):
                if tuple(words[pos:pos + SHINGLE]) in shared:
                    mask[pos:pos + SHINGLE] = ONES

    return {pair: mask.count(1) / len(mask) for pair, mask in coverage.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing Markdown files (recursive)")
    parser.add_argument("--min-score", type=float, default=0.10,
                        help="Minimum document-pair reuse score, 0–1 (default: 0.10)")
    parser.add_argument("--window-score", type=float, default=0.35,
                        help="Minimum shared-shingle containment within local windows (default: 0.35)")
    parser.add_argument("--shared", type=int, default=20,
                        help="Minimum shared six-word shingles per window (default: 20)")
    parser.add_argument("--max-frequency", type=int, default=100,
                        help="Ignore shingles in more than this many windows (default: 100)")
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"Not a directory: {args.directory}")
    if not 0 <= args.min_score <= 1 or not 0 <= args.window_score <= 1:
        parser.error("Scores must be between 0 and 1")

    paths = sorted(args.directory.rglob("*.md"))
    documents = [read_words(path) for path in paths]
    scores = find_scores(documents, make_windows(documents),
                         args.window_score, args.shared, args.max_frequency)

    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(("file_a", "file_b", "score"))
    for (a, b), score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        if score >= args.min_score:
            writer.writerow((paths[a].relative_to(args.directory).with_suffix(""),
                 paths[b].relative_to(args.directory).with_suffix(""),
                 f"{score:.3f}"))


if __name__ == "__main__":
    main()
