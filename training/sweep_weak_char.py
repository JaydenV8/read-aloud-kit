#!/usr/bin/env python3
"""Derive the weak-character threshold from this corpus.

    training/.venv/bin/python training/sweep_weak_char.py

`WEAK_CHAR_GOP` in @readaloudkit/gop decides when a character inside a word
counts as weak, which feeds `nWeakChars`, `fracWeakChars` and two utterance
aggregates. A threshold is a fitted constant like any other, so it has to come
from the corpus the model ships against — inheriting one measured elsewhere
would put an unattributable number in the weights.

The sweep asks: at threshold t, how well does "this word contains a character
below t" predict "the expert marked this word down"? The answer is the value
maximising F1 on the **training** split. The test split is never read here; a
threshold selected against it would leak it into every downstream head.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VAL_SPEAKER_FRACTION = 0.2


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=HERE / "data" / "features.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "weak_char_sweep.json")
    ap.add_argument("--lo", type=float, default=-6.0)
    ap.add_argument("--hi", type=float, default=-0.2)
    ap.add_argument("--step", type=float, default=0.1)
    args = ap.parse_args()

    rows = load(args.features)
    if not rows:
        print("no features; run extract_features.ts first", file=sys.stderr)
        return 1

    # Train split only, and not even the speakers train.py holds out for
    # validation — those pick early stopping, so a threshold tuned on them
    # would be selected against the same data twice.
    speakers = sorted({r["speakerId"] for r in rows if r["split"] == "train"})
    n_val = max(1, int(round(len(speakers) * VAL_SPEAKER_FRACTION)))
    val_speakers = set(speakers[:: max(1, len(speakers) // n_val)][:n_val])

    series: list[list[float]] = []
    marked: list[int] = []
    for r in rows:
        if r["split"] != "train" or r["speakerId"] in val_speakers:
            continue
        for w in r["words"]:
            gops = w.get("charGops")
            if not gops:
                continue
            series.append(gops)
            marked.append(0 if w["level"] == "good" else 1)

    if not series:
        print("features carry no charGops; re-run extract_features.ts", file=sys.stderr)
        return 1

    y = np.asarray(marked)
    mins = np.asarray([min(s) for s in series])
    print(f"{len(y)} training words, {y.mean():.1%} marked down by the expert")
    print(f"per-character posterior, word minimum: "
          f"p05={np.quantile(mins, 0.05):.2f}  median={np.median(mins):.2f}  p95={np.quantile(mins, 0.95):.2f}\n")

    best = None
    table = []
    for t in np.arange(args.lo, args.hi + 1e-9, args.step):
        pred = (mins <= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        table.append({"threshold": round(float(t), 3), "precision": p, "recall": r, "f1": f1})
        if best is None or f1 > best["f1"]:
            best = table[-1]

    assert best is not None
    print(f"{'threshold':>10} {'precision':>10} {'recall':>8} {'F1':>8}")
    for row in table:
        mark = "  <-- best" if row["threshold"] == best["threshold"] else ""
        if abs(row["threshold"] * 10) % 5 < 1e-6 or mark:
            print(f"{row['threshold']:>10.1f} {row['precision']:>10.3f} {row['recall']:>8.3f}"
                  f" {row['f1']:>8.3f}{mark}")

    print(f"\nWEAK_CHAR_GOP = {best['threshold']}")
    print("Set this in packages/gop/src/index.ts and re-run extract_features.ts.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "selected": best,
                "swept_on": "speechocean762 train split, minus the speakers train.py holds out",
                "n_words": int(len(y)),
                "positive_rate": float(y.mean()),
                "grid": table,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
