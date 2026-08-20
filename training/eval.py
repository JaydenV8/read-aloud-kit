#!/usr/bin/env python3
"""Evaluate the exported heads on the corpus test split.

    training/.venv/bin/python training/eval.py

This loads the ONNX graphs rather than the LightGBM models, so it measures what
actually ships. Anyone with the corpus can run it and reproduce the numbers in
MODEL_CARD.md without needing the training configuration.
"""

from __future__ import annotations

import argparse
import re
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

HIDDEN_KEY = re.compile(r"^hid\d+$")
HERE = Path(__file__).resolve().parent


def load_rows(path: Path, split: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if split == "all" or row["split"] == split:
                rows.append(row)
    return rows


def run(session: ort.InferenceSession, x: np.ndarray) -> list[np.ndarray]:
    return session.run(None, {session.get_inputs()[0].name: x.astype(np.float32)})


def pick_proba(outs: list[np.ndarray], n_classes: int) -> np.ndarray:
    for arr in outs:
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[1] == n_classes:
            return a
    raise SystemExit("classifier graph has no probability output")


def decide(proba: np.ndarray, levels: list[str], threshold: float) -> np.ndarray:
    """The rule the runtime applies, not a bare argmax.

    A word is called out only when the probability that it is not `good` clears
    the threshold; the band within a flagged word is then the stronger of the
    remaining classes. Deriving both readings from one number keeps a word from
    being shown as `bad` while the flag says it is fine, and roughly doubles the
    precision of the call — see ATTENTION_THRESHOLD in @readaloudkit/scoring.
    """
    good = levels.index("good")
    not_good_cols = [i for i in range(len(levels)) if i != good]
    not_good = proba[:, not_good_cols].sum(axis=1)
    sub = proba[:, not_good_cols].argmax(axis=1)
    picked = np.asarray(not_good_cols, dtype=np.int64)[sub]
    return np.where(not_good >= threshold, picked, good)


def pearson(y: np.ndarray, p: np.ndarray) -> float:
    if y.size < 3 or np.std(y) < 1e-9 or np.std(p) < 1e-9:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=HERE / "data" / "features.jsonl")
    ap.add_argument("--onnx", type=Path, default=HERE / "artifacts" / "onnx")
    ap.add_argument("--split", default="test", choices=("train", "test", "all"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--hidden", type=Path, default=HERE / "data" / "hidden_l3.f32")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="must match ATTENTION_THRESHOLD in @readaloudkit/scoring",
    )
    args = ap.parse_args()

    manifest = json.loads((args.onnx / "scoring.json").read_text(encoding="utf-8"))
    levels = manifest["wordLevels"]
    utt_keys = manifest["utteranceFeatureKeys"]
    meta = json.loads((HERE / "data" / "features.meta.json").read_text(encoding="utf-8"))

    # Which extraction the shipped weights read. The heads carry their own key
    # list, so the release decides this, not a flag: evaluating v2 weights
    # against v1 columns would silently score a different model.
    version = int(manifest.get("provenance", {}).get("featureVersion", 1))
    word_field = "featuresV2" if version >= 2 else "features"
    utt_field = "featuresV2" if version >= 2 else "features"
    all_keys = meta["utteranceFeatureKeysV2" if version >= 2 else "utteranceFeatureKeys"]
    # `hid*` come from the release's own projection of the pooled hidden layer,
    # not from the extraction, so they are not looked for among the feature keys.
    hidden_manifest = manifest.get("hiddenProjection")
    n_hidden = hidden_manifest["components"] if hidden_manifest else 0
    utt_keys = [k for k in utt_keys if not HIDDEN_KEY.match(k)]
    missing = [k for k in utt_keys if k not in all_keys]
    if missing:
        raise SystemExit(f"features lack {missing}; re-run extract_features.ts")
    keep = [all_keys.index(k) for k in utt_keys]
    print(f"release {manifest.get('version', '?')}, feature version {version}")

    rows = load_rows(args.features, args.split)
    if not rows:
        raise SystemExit(f"no rows for split {args.split}")
    print(f"{len(rows)} utterances in {args.split}")

    word_hidden = utt_hidden = None
    if n_hidden:
        # Evaluated through the same graph the runtime multiplies by, so this
        # measures the projection that ships rather than the fit it came from.
        all_rows = load_rows(args.features, "all")
        flat = np.fromfile(args.hidden, dtype=np.float32).reshape(-1, 768)
        want = sum(len(r["words"]) + 1 for r in all_rows)
        if flat.shape[0] != want:
            raise SystemExit(f"{args.hidden} holds {flat.shape[0]} rows, features imply {want}")
        keep_ids = {r["utteranceId"] for r in rows}
        w_sel, u_sel, off = [], [], 0
        for r in all_rows:
            n = len(r["words"])
            if r["utteranceId"] in keep_ids:
                w_sel.append(flat[off : off + n])
                u_sel.append(flat[off + n])
            off += n + 1
        proj = lambda name, X: ort.InferenceSession(  # noqa: E731
            str(args.onnx / hidden_manifest[name]), providers=["CPUExecutionProvider"]
        ).run(None, {"hidden": X})[0]
        word_hidden = proj("word", np.vstack(w_sel))
        utt_hidden = proj("utterance", np.vstack(u_sel))
        print(f"hidden layer {hidden_manifest['layer']}: +{n_hidden} features")

    report: dict = {"split": args.split, "utterances": len(rows), "threshold": args.threshold}

    word_x = np.asarray([w[word_field] for r in rows for w in r["words"]], dtype=np.float32)
    if word_hidden is not None:
        word_x = np.hstack([word_x, word_hidden])
    word_y = np.asarray(
        [levels.index(w["level"]) for r in rows for w in r["words"]], dtype=np.int64
    )
    sess = ort.InferenceSession(str(args.onnx / "word_level.onnx"), providers=["CPUExecutionProvider"])
    proba = pick_proba(run(sess, word_x), len(levels))
    pred = decide(proba, levels, args.threshold)

    per_class = {}
    for i, name in enumerate(levels):
        tp = int(((pred == i) & (word_y == i)).sum())
        fp = int(((pred == i) & (word_y != i)).sum())
        fn = int(((pred != i) & (word_y == i)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int((word_y == i).sum()),
        }
    good_i = levels.index("good")
    flagged = pred != good_i
    truly = word_y != good_i
    tp = int((flagged & truly).sum())
    fp = int((flagged & ~truly).sum())
    attention = {
        "flagged": int(flagged.sum()),
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / int(truly.sum()), 4) if truly.sum() else 0.0,
        "trueRate": round(float(truly.mean()), 4),
    }
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
    majority = float(np.bincount(word_y).max() / word_y.size)
    report["word_level"] = {
        "n": int(word_y.size),
        "accuracy": round(float((pred == word_y).mean()), 4),
        "macro_f1": round(macro_f1, 4),
        "majority_class_accuracy": round(majority, 4),
        "per_class": per_class,
        "needs_attention": attention,
    }
    print(f"\nword_level  n={word_y.size}")
    print(f"  accuracy {report['word_level']['accuracy']:.3f}  (all-good baseline {majority:.3f})")
    print(f"  macro F1 {macro_f1:.3f}")
    for name, v in per_class.items():
        print(f"    {name:8} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} n={v['support']}")
    print(
        f"  needsAttention: flagged {attention['flagged']}  P={attention['precision']:.3f} "
        f"R={attention['recall']:.3f}  (true rate {attention['trueRate']:.3f})"
    )

    utt_x = np.asarray(
        [[r["utterance"][utt_field][i] for i in keep] for r in rows], dtype=np.float32
    )
    if utt_hidden is not None:
        utt_x = np.hstack([utt_x, utt_hidden])
    print()
    for head, target in (
        ("utterance_accuracy", "accuracy"),
        ("utterance_fluency", "fluency"),
        ("utterance_total", "total"),
    ):
        path = args.onnx / f"{head}.onnx"
        if not path.exists():
            continue
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        pred = np.asarray(run(sess, utt_x)[0], dtype=np.float64).reshape(-1)
        y = np.asarray([r["utterance"]["labels"][target] for r in rows], dtype=np.float64)
        report[head] = {
            "n": int(y.size),
            "pearson": round(pearson(y, pred), 4),
            "mae": round(float(np.mean(np.abs(y - pred))), 4),
            "predict_mean_mae": round(float(np.mean(np.abs(y - y.mean()))), 4),
        }
        s = report[head]
        print(
            f"{head:20} r={s['pearson']:.3f}  mae={s['mae']:.3f}  "
            f"(predicting the mean gives {s['predict_mean_mae']:.3f})"
        )

    out = args.out or (args.onnx / f"eval-{args.split}.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
