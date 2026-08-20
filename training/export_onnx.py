#!/usr/bin/env python3
"""Export trained heads to ONNX for the Node runtime.

    training/.venv/bin/python training/export_onnx.py

Every graph is run through onnxruntime and compared against the LightGBM model
it came from. An export that silently changes predictions is worse than no
export, so a mismatch aborts rather than warns.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
INPUT_NAME = "input"

# The corpus scores 0-10. The published Read Aloud report is 10-90, and readers
# expect that range, so the runtime rescales linearly. This is a presentation
# mapping onto a familiar range, fitted to nothing: it does not make the number
# an official score, and the model card has to say so wherever it appears.
DISPLAY_SCALE = {"from": [0.0, 10.0], "to": [10.0, 90.0]}

# word_stress is exported for reproducibility but not loaded by the runtime.
# At 0.044 precision on the test split it raises about twenty false alarms per
# real one, which is worse than saying nothing.
SHIPPED = ("word_level", "utterance_accuracy", "utterance_fluency", "utterance_total")

CLASSIFIER_TOL = 2e-4
REGRESSOR_TOL = 2e-3


def convert(model, n_features: int, kind: str):
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    booster = model.booster_ if hasattr(model, "booster_") else model
    kwargs = {
        "initial_types": [(INPUT_NAME, FloatTensorType([None, n_features]))],
        "target_opset": 15,
    }
    if kind == "classifier":
        kwargs["zipmap"] = False
    return convert_lightgbm(booster, **kwargs)


def check(path: Path, model, kind: str, n_features: int) -> dict:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, n_features)).astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {sess.get_inputs()[0].name: x})

    if kind == "classifier":
        expected = np.asarray(model.predict_proba(x), dtype=np.float32)
        got = None
        for arr in outs:
            a = np.asarray(arr)
            if a.ndim == 2 and a.shape[1] == expected.shape[1]:
                got = a.astype(np.float32)
        if got is None:
            raise SystemExit(f"{path.name}: the graph has no probability output")
        max_abs = float(np.max(np.abs(got - expected)))
        tol = CLASSIFIER_TOL
    else:
        expected = np.asarray(model.predict(x), dtype=np.float32).reshape(-1)
        got = np.asarray(outs[0], dtype=np.float32).reshape(-1)
        max_abs = float(np.max(np.abs(got - expected)))
        tol = REGRESSOR_TOL

    if max_abs >= tol:
        raise SystemExit(f"{path.name}: onnx differs from lightgbm by {max_abs:.3e} (tol {tol:.0e})")
    return {"maxAbs": max_abs, "tolerance": tol}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "onnx")
    args = ap.parse_args()

    bundle = joblib.load(args.artifacts / "heads.joblib")
    metrics = json.loads((args.artifacts / "metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((args.artifacts / "provenance.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    exported = {}
    for name, model in bundle["models"].items():
        kind = "classifier" if name.startswith("word_") else "regressor"
        n = int(model.n_features_in_)
        graph = convert(model, n, kind)
        path = args.out / f"{name}.onnx"
        path.write_bytes(graph.SerializeToString())
        parity = check(path, model, kind, n)
        exported[name] = {
            "file": f"{name}.onnx",
            "kind": kind,
            "nFeatures": n,
            "bytes": path.stat().st_size,
            "shipped": name in SHIPPED,
            "parity": parity,
        }
        flag = "" if name in SHIPPED else "   (not loaded by the runtime)"
        print(f"  {name:22} {path.stat().st_size / 1024:7.1f} KB  parity={parity['maxAbs']:.2e}{flag}")

    # The number tracks the feature contract the heads were fitted on: 0.4 is the
    # eleven-average word summary, 0.5 keeps the per-character series, 0.6 adds a
    # projection of an intermediate acoustic layer. The suffix marks the corpus,
    # and every release here is speechocean762 and nothing else.
    version = f"0.{ {1: 4, 2: 5, 3: 6}.get(provenance.get('featureVersion', 1), 4) }-community"

    # The projection is part of the release, not of the acoustic model: it is
    # fitted on this corpus, while the checkpoint stays a stock third-party
    # export that happens to expose one more tensor.
    hidden = provenance.get("hidden")
    hidden_manifest = None
    if hidden:
        for name in ("word", "utterance"):
            src = args.artifacts / hidden["fits"][name]["file"]
            shutil.copyfile(src, args.out / src.name)
            print(f"  {'hidden_' + name:22} {src.stat().st_size / 1024:7.1f} KB  "
                  f"evr={hidden['fits'][name]['explainedVariance']:.3f}")
        hidden_manifest = {
            "layer": hidden["layer"],
            "word": hidden["fits"]["word"]["file"],
            "utterance": hidden["fits"]["utterance"]["file"],
            "size": hidden["size"],
            "components": hidden["components"],
        }

    manifest = {
        "version": version,
        "inputName": INPUT_NAME,
        "wordFeatureKeys": bundle["word_keys"],
        "utteranceFeatureKeys": bundle["utterance_keys"],
        **({"hiddenProjection": hidden_manifest} if hidden_manifest else {}),
        "wordLevels": list(bundle["word_levels"]),
        "displayScale": DISPLAY_SCALE,
        "heads": exported,
        "outputs": {
            "words[].level": "word_level",
            "scores.pronunciation": "utterance_accuracy",
            "scores.fluency": "utterance_fluency",
            "scores.overall": "utterance_total",
        },
        "metrics": {k: v.get("test") for k, v in metrics.items() if isinstance(v, dict)},
        "metricsNote": (
            "Raw argmax over the head's classes. The runtime does not score this way: it collapses to the ATTENTION_THRESHOLD decision, which MODEL_CARD.md reports and training/eval.py reproduces. Expect the card's accuracy to be higher than the number here — they are different rules, not different models."
        ),
        "provenance": provenance,
    }
    (args.out / "scoring.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total = sum(v["bytes"] for k, v in exported.items() if v["shipped"])
    print(f"\nwrote {args.out}   shipped heads total {total / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
