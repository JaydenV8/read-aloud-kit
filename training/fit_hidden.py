#!/usr/bin/env python
"""Fit the hidden-layer projection and export it as ONNX.

The scoring heads cannot take 768 columns of pooled transformer output -- on
2000 training utterances that is a way of memorising speakers. This reduces it
to a handful of components, fitted on the training split only, and writes the
result as a graph the runtime multiplies by.

Standardisation and PCA are both affine, so they fold into one `y = xA + b`:

    y = ((x - mu) / sigma) W^T  =  x (W / sigma)^T - (mu / sigma) W^T

which is a single Gemm and exactly equal to running the two in sequence, not an
approximation of it. Words and whole clips get separate fits: they are pooled
over different spans and their covariance is not the same.

    python training/fit_hidden.py --components 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
HIDDEN_SIZE = 768


def load(features: Path, sidecar: Path) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    rows = [json.loads(l) for l in features.open(encoding="utf-8") if l.strip()]
    flat = np.fromfile(sidecar, dtype=np.float32)
    if flat.size % HIDDEN_SIZE:
        raise SystemExit(f"{sidecar}: {flat.size} floats is not a multiple of {HIDDEN_SIZE}")
    flat = flat.reshape(-1, HIDDEN_SIZE)
    want = sum(len(r["words"]) + 1 for r in rows)
    if flat.shape[0] != want:
        raise SystemExit(f"{sidecar} holds {flat.shape[0]} rows, the JSONL implies {want}")

    words, utts, word_of = [], [], []
    off = 0
    for i, r in enumerate(rows):
        n = len(r["words"])
        words.append(flat[off : off + n])
        utts.append(flat[off + n])
        word_of.extend([i] * n)
        off += n + 1
    return rows, np.vstack(words), np.vstack(utts), np.asarray(word_of)


def fold(scaler: StandardScaler, pca: PCA) -> tuple[np.ndarray, np.ndarray]:
    sigma = scaler.scale_.astype(np.float64)
    mu = scaler.mean_.astype(np.float64)
    W = pca.components_.astype(np.float64)
    centre = pca.mean_.astype(np.float64)
    # PCA subtracts its own mean of the standardised data as well.
    A = (W / sigma).T
    b = -((mu / sigma) + centre) @ W.T
    return A.astype(np.float32), b.astype(np.float32)


def export(A: np.ndarray, b: np.ndarray, path: Path) -> None:
    from onnx import TensorProto, helper, numpy_helper, save

    node = helper.make_node(
        "Gemm", ["hidden", "A", "b"], ["z"], alpha=1.0, beta=1.0, transA=0, transB=0
    )
    graph = helper.make_graph(
        [node],
        "hidden_projection",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, ["n", A.shape[0]])],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, ["n", A.shape[1]])],
        [numpy_helper.from_array(A, "A"), numpy_helper.from_array(b, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    save(model, str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=HERE / "data" / "features_v3.jsonl")
    ap.add_argument("--hidden", type=Path, default=HERE / "data" / "hidden_l3.f32")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts")
    ap.add_argument("--components", type=int, default=32)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows, words, utts, word_of = load(args.features, args.hidden)
    train_utt = np.asarray([r["split"] != "test" for r in rows])
    train_word = train_utt[word_of]
    print(f"{len(rows)} utterances, {words.shape[0]} words")
    print(f"fitting on {int(train_utt.sum())} utterances / {int(train_word.sum())} words "
          f"(test held out entirely)")

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"layer": args.layer, "size": HIDDEN_SIZE, "components": args.components, "fits": {}}
    for name, X, mask in (("word", words, train_word), ("utterance", utts, train_utt)):
        sc = StandardScaler().fit(X[mask])
        pca = PCA(n_components=args.components, random_state=args.seed).fit(sc.transform(X[mask]))
        A, b = fold(sc, pca)

        # The fold has to be exact, not close: it is applied to every request
        # while the sequential form is only ever run here.
        ref = pca.transform(sc.transform(X[:2000]))
        got = X[:2000] @ A + b
        err = float(np.abs(ref - got).max())
        scale = float(np.abs(ref).max())
        print(f"  {name:<10} evr {pca.explained_variance_ratio_.sum():.4f}   "
              f"fold error {err:.3e} ({err / scale:.1e} rel)")
        if err / scale > 1e-5:
            raise SystemExit(f"{name}: folded projection does not match the sequential one")

        path = args.out / f"hidden_{name}.onnx"
        export(A, b, path)

        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"hidden": X[:2000]})[0]
        ort_err = float(np.abs(ort_out - ref).max())
        print(f"  {'':<10} onnx {ort_err:.3e} ({ort_err / scale:.1e} rel)  -> {path.name}")
        if ort_err / scale > 1e-5:
            raise SystemExit(f"{name}: ONNX projection disagrees with the fit")
        meta["fits"][name] = {
            "file": path.name,
            "explainedVariance": float(pca.explained_variance_ratio_.sum()),
            "foldError": err,
            "onnxError": ort_err,
            "fittedOn": int(mask.sum()),
        }

    (args.out / "hidden_projection.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"-> {args.out / 'hidden_projection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
