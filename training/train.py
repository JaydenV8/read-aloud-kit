#!/usr/bin/env python3
"""Train scoring heads on extracted features.

    training/.venv/bin/python training/train.py

Every head is a LightGBM model over the feature vectors written by
extract_features.ts, which is the same code the HTTP API runs.

Provenance rules, asserted rather than trusted:

  * labels come only from the corpora listed in the manifest
  * validation and early stopping use speakers held out of the training split,
    never the official test split — selecting against test leaks it into the
    weights as surely as training on it would
  * no constant fitted anywhere else is copied in
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

HERE = Path(__file__).resolve().parent
SEED = 42

WORD_LEVELS = ("good", "average", "bad")
WORD_LEVEL_ID = {k: i for i, k in enumerate(WORD_LEVELS)}

# Wrong stress is coded 5 and correct 10, and mono-syllable words are always
# correct by definition, so the positive class is both rare and only reachable
# in polysyllables.
STRESS_WRONG = 5.0

# Utterance features whose magnitude tracks how long the prompt is. On this
# corpus they are the strongest fluency predictors — a hesitant reader takes
# longer over six words. On the 40-80 word prompts this project targets, the
# same features are dominated by prompt length instead, so a head that leaned
# on them would not transfer. Trained without them by default; --utterance-
# features all measures what that costs.
# `gapBeforeMax` and `wordGopRange` belong here for the same reason one step
# removed: both are extremes over the word count, so they drift upward simply
# because a longer prompt offers more chances to hit one.
LENGTH_SCALING = frozenset(
    {
        "duration",
        "nRef",
        "nAligned",
        "spokenSec",
        "nPause",
        "pauseTotal",
        "nGopWords",
        "gapBeforeMax",
        "wordGopRange",
    }
)

# Signal level, not speech quality. Every speaker here was recorded once, so
# recording conditions track the speaker, and the speaker tracks proficiency —
# a head that leaned on gain would look good for the wrong reason. Removing all
# four costs nothing measurable (accuracy 0.660 -> 0.652, fluency 0.733 ->
# 0.729, prosodic and total both improve), so they are out by default.
# `rmsStd` is in any case the same number as `rmsMean` to nine decimal places:
# they differ only by the signal mean, and audio is DC-free.
RECORDING_LEVEL = frozenset({"rmsMean", "rmsStd", "rmsPeak", "activeRatio"})

UTTERANCE_TARGETS = ("accuracy", "fluency", "prosodic", "total")
VAL_SPEAKER_FRACTION = 0.2


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def speaker_val_split(rows: list[dict]) -> set[str]:
    """Hold out whole speakers. Splitting by utterance would let the same voice
    appear on both sides and make validation look better than it is."""
    speakers = sorted({r["speakerId"] for r in rows if r["split"] == "train"})
    rng = np.random.default_rng(SEED)
    rng.shuffle(speakers)
    n = max(1, int(len(speakers) * VAL_SPEAKER_FRACTION))
    return set(speakers[:n])


def balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (len(counts) * counts)
    return w[y]


def fit_classifier(Xtr, ytr, Xva, yva, n_classes: int):
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        num_class=n_classes if n_classes > 2 else None,
    )
    model.fit(
        Xtr,
        ytr,
        sample_weight=balanced_weights(ytr),
        eval_X=Xva,
        eval_y=yva,
        eval_sample_weight=[balanced_weights(yva)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def fit_regressor(Xtr, ytr, Xva, yva):
    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def pearson(y: np.ndarray, p: np.ndarray) -> float:
    if y.size < 3 or np.std(y) < 1e-9 or np.std(p) < 1e-9:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def regression_report(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": int(y.size),
        "pearson": round(pearson(y, p), 4),
        "mae": round(float(np.mean(np.abs(y - p))), 4),
        "rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 4),
        "predict_mean_mae": round(float(np.mean(np.abs(y - np.mean(y)))), 4),
    }


def classification_summary(y: np.ndarray, pred: np.ndarray, names: tuple[str, ...]) -> dict:
    labels = list(range(len(names)))
    rep = classification_report(
        y, pred, labels=labels, target_names=list(names), output_dict=True, zero_division=0
    )
    return {
        "n": int(y.size),
        "accuracy": round(float((y == pred).mean()), 4),
        "macro_f1": round(
            float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)), 4
        ),
        "per_class": {
            k: {
                "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4),
                "f1": round(v["f1-score"], 4),
                "support": int(v["support"]),
            }
            for k, v in rep.items()
            if k in names
        },
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
        "majority_class_accuracy": round(float(Counter(y.tolist()).most_common(1)[0][1] / y.size), 4),
    }


def build_word_sets(rows: list[dict], val_speakers: set[str], field: str = "features"):
    packs = {"train": [], "val": [], "test": []}
    for r in rows:
        bucket = "test" if r["split"] == "test" else ("val" if r["speakerId"] in val_speakers else "train")
        for w in r["words"]:
            packs[bucket].append(w)
    out = {}
    for name, words in packs.items():
        out[name] = {
            "X": np.asarray([w[field] for w in words], dtype=np.float32),
            "level": np.asarray([WORD_LEVEL_ID[w["level"]] for w in words], dtype=np.int32),
            "stress": np.asarray(
                [1 if w["stress"] <= STRESS_WRONG else 0 for w in words], dtype=np.int32
            ),
        }
    return out


def build_utterance_sets(
    rows: list[dict], val_speakers: set[str], keep: list[int], field: str = "features"
):
    packs = {"train": [], "val": [], "test": []}
    for r in rows:
        bucket = "test" if r["split"] == "test" else ("val" if r["speakerId"] in val_speakers else "train")
        packs[bucket].append(r)
    out = {}
    for name, rs in packs.items():
        out[name] = {
            "X": np.asarray([[r["utterance"][field][i] for i in keep] for r in rs], dtype=np.float32),
            **{
                t: np.asarray([r["utterance"]["labels"][t] for r in rs], dtype=np.float32)
                for t in UTTERANCE_TARGETS
            },
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=HERE / "data" / "features.jsonl")
    ap.add_argument("--meta", type=Path, default=HERE / "data" / "features.meta.json")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts")
    ap.add_argument("--adults-only", action="store_true")
    ap.add_argument(
        "--utterance-features",
        choices=("invariant", "all"),
        default="invariant",
        help="invariant drops length-scaling and recording-level features",
    )
    ap.add_argument(
        "--drop",
        default="",
        help="comma-separated utterance features to exclude, for ablations",
    )
    ap.add_argument(
        "--feature-version",
        type=int,
        choices=(1, 2),
        default=2,
        help="1 summarises each word as eleven averages; 2 keeps the shape of "
        "its per-character series. Both read the same extraction.",
    )
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    if args.feature_version == 2:
        if "wordFeatureKeysV2" not in meta:
            print("features predate v2; re-run extract_features.ts", file=sys.stderr)
            return 1
        field = "featuresV2"
        word_keys = list(meta["wordFeatureKeysV2"])
        utt_keys_all = list(meta["utteranceFeatureKeysV2"])
    else:
        field = "features"
        word_keys = list(meta["wordFeatureKeys"])
        utt_keys_all = list(meta["utteranceFeatureKeys"])

    rows = load_rows(args.features)
    if args.adults_only:
        rows = [r for r in rows if r.get("adult")]
    if not rows:
        print("no features; run extract_features.ts first", file=sys.stderr)
        return 1

    dropped = {k.strip() for k in args.drop.split(",") if k.strip()}
    unknown = dropped - set(utt_keys_all)
    if unknown:
        print(f"--drop names features that do not exist: {sorted(unknown)}", file=sys.stderr)
        return 1
    keep_idx = [
        i
        for i, k in enumerate(utt_keys_all)
        if (args.utterance_features == "all" or k not in (LENGTH_SCALING | RECORDING_LEVEL))
        and k not in dropped
    ]
    utt_keys = [utt_keys_all[i] for i in keep_idx]

    val_speakers = speaker_val_split(rows)
    train_speakers = {r["speakerId"] for r in rows if r["split"] == "train"} - val_speakers
    test_speakers = {r["speakerId"] for r in rows if r["split"] == "test"}
    assert not (val_speakers & test_speakers), "validation speakers leaked from test"
    assert not (train_speakers & test_speakers), "training speakers leaked from test"

    words = build_word_sets(rows, val_speakers, field)
    utts = build_utterance_sets(rows, val_speakers, keep_idx, field)
    print(
        f"utterances train/val/test = {len(utts['train']['X'])}/{len(utts['val']['X'])}/{len(utts['test']['X'])}"
        f"   words = {len(words['train']['X'])}/{len(words['val']['X'])}/{len(words['test']['X'])}"
    )
    print(f"utterance features: {len(utt_keys)} of {len(utt_keys_all)} ({args.utterance_features})")

    models: dict[str, object] = {}
    metrics: dict[str, dict] = {}

    print("\nword_level (good / average / bad)")
    m = fit_classifier(
        words["train"]["X"], words["train"]["level"], words["val"]["X"], words["val"]["level"], 3
    )
    models["word_level"] = m
    for split in ("val", "test"):
        pred = m.predict(words[split]["X"])
        metrics.setdefault("word_level", {})[split] = classification_summary(
            words[split]["level"], np.asarray(pred), WORD_LEVELS
        )
        s = metrics["word_level"][split]
        print(f"  {split:5} acc={s['accuracy']:.3f} macroF1={s['macro_f1']:.3f} (majority {s['majority_class_accuracy']:.3f})")

    print("\nword_stress (wrong stress = positive)")
    m = fit_classifier(
        words["train"]["X"], words["train"]["stress"], words["val"]["X"], words["val"]["stress"], 2
    )
    models["word_stress"] = m
    for split in ("val", "test"):
        pred = np.asarray(m.predict(words[split]["X"]))
        metrics.setdefault("word_stress", {})[split] = classification_summary(
            words[split]["stress"], pred, ("correct", "wrong")
        )
        s = metrics["word_stress"][split]
        wrong = s["per_class"]["wrong"]
        print(
            f"  {split:5} acc={s['accuracy']:.3f} wrong: P={wrong['precision']:.3f} "
            f"R={wrong['recall']:.3f} F1={wrong['f1']:.3f} (n={wrong['support']})"
        )

    for target in UTTERANCE_TARGETS:
        print(f"\nutterance_{target}")
        m = fit_regressor(
            utts["train"]["X"], utts["train"][target], utts["val"]["X"], utts["val"][target]
        )
        models[f"utterance_{target}"] = m
        for split in ("val", "test"):
            pred = m.predict(utts[split]["X"])
            metrics.setdefault(f"utterance_{target}", {})[split] = regression_report(
                utts[split][target], np.asarray(pred)
            )
            s = metrics[f"utterance_{target}"][split]
            print(
                f"  {split:5} r={s['pearson']:.3f} mae={s['mae']:.3f} "
                f"(predicting the mean would give mae={s['predict_mean_mae']:.3f})"
            )
        imp = sorted(zip(utt_keys, m.feature_importances_), key=lambda kv: -kv[1])[:5]
        metrics[f"utterance_{target}"]["top_features"] = [[k, int(v)] for k, v in imp]
        print(f"  leans on: {', '.join(k for k, _ in imp)}")

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "word_keys": word_keys,
            "utterance_keys": utt_keys,
            "word_levels": WORD_LEVELS,
        },
        args.out / "heads.joblib",
    )
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    corpus_lock = json.loads((HERE / "corpus.lock.json").read_text(encoding="utf-8"))
    provenance = {
        "corpora": [
            {
                "name": "speechocean762",
                "license": "CC BY 4.0",
                "url": corpus_lock.get("url"),
                "sha256": corpus_lock.get("sha256"),
                "citation": "speechocean762: An Open-Source Non-Native English Speech Corpus "
                "for Pronunciation Assessment. Interspeech 2021.",
            }
        ],
        "seed": SEED,
        "split": {
            "test": "corpus official test split, untouched during training",
            "val": f"{len(val_speakers)} speakers held out of the official train split",
            "trainSpeakers": len(train_speakers),
            "valSpeakers": len(val_speakers),
            "testSpeakers": len(test_speakers),
        },
        "filters": {"adultsOnly": args.adults_only},
        "featureVersion": args.feature_version,
        # Every fitted constant has to be attributable, including this one.
        "weakCharGop": meta.get("weakCharGop"),
        "utteranceFeatures": {
            "mode": args.utterance_features,
            "dropped": sorted(dropped),
            "keys": utt_keys,
        },
        "wordFeatures": word_keys,
        "toolchain": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
        },
    }
    (args.out / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
