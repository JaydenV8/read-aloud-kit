#!/usr/bin/env python
"""Export the acoustic model to ONNX.

The shipped graph emits CTC logits and nothing else, which is everything the
alignment and the decode need and, it turns out, less than the audio contains.
A logit layer is trained to discard whatever does not help pick a character;
measured against expert labels, the last transformer layer is the *least* useful
of the twelve to the scoring heads and an early one is the most useful. So this
exports a chosen hidden layer alongside the logits.

Both outputs come off one forward pass. The hidden layer is a tensor the graph
already computes on its way to the logits, so emitting it costs no weights, no
second session and no extra arithmetic -- only the copy out.

    python training/export_acoustic.py --out models/wav2vec2-base-960h-ctc.onnx

The layer is indexed from 1, matching `extract_features()[0][n - 1]`. Parity
against the PyTorch pipeline is checked on real audio and the export is refused
if it drifts, because a silently wrong hidden layer would not break the decode
-- it would just quietly make the scores worse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
BUNDLE = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
# Chosen by 5-fold speaker-disjoint cross-validation over the training half of
# speechocean762, refitting the projection inside each fold. Both the word head
# and the utterance heads picked it independently. See docs/models.md.
DEFAULT_HIDDEN_LAYER = 3
# Parity is judged relative to each tensor's own scale. The shipped export
# recorded an absolute 5.0e-05, but that was one 49-frame clip: drift
# accumulates over the sequence, and the same graph is 5.1e-03 on a 128-frame
# one. Absolute tolerance therefore encodes the length of whatever clip it was
# set on. Logits run to +-17 here, so 5.1e-03 is 3e-04 of full scale, and the
# argmax -- the only thing the decode reads -- is unchanged.
TOLERANCE = 1e-3


class AcousticExport(nn.Module):
    """Wav2Vec2Model's own forward, with one intermediate tensor kept.

    Written out rather than hooked so that what the graph emits is legible here.
    Every line mirrors torchaudio's `Wav2Vec2Model.forward` /
    `Encoder.forward` / `Transformer.forward`; `test_matches_pipeline` below is
    what actually holds them together.
    """

    def __init__(self, model: nn.Module, layer: int) -> None:
        super().__init__()
        self.model = model
        n = len(model.encoder.transformer.layers)
        if not 1 <= layer <= n:
            raise SystemExit(f"--hidden-layer must be in [1, {n}], got {layer}")
        self.index = layer - 1

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        x, _ = self.model.feature_extractor(waveform, None)
        # No lengths: a batch of one, so there is nothing to mask.
        x, mask = self.model.encoder._preprocess(x, None)
        t = self.model.encoder.transformer
        x = t._preprocess(x)
        hidden = x
        position_bias = None
        for i, layer in enumerate(t.layers):
            x, position_bias = layer(x, mask, position_bias=position_bias)
            if i == self.index:
                hidden = x
        if not t.layer_norm_first:
            x = t.layer_norm(x)
        return self.model.aux(x), hidden


def read_wav(path: Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit PCM")
        if w.getframerate() != BUNDLE.sample_rate:
            raise SystemExit(f"{path}: expected {BUNDLE.sample_rate} Hz")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        pcm = pcm.astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(1)
    return torch.from_numpy(pcm.copy()).unsqueeze(0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parent / "models" / "wav2vec2-base-960h-ctc.onnx")
    ap.add_argument("--hidden-layer", type=int, default=DEFAULT_HIDDEN_LAYER)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check-audio", type=Path, action="append", default=None)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()
    clips = args.check_audio or [HERE.parent / "examples" / "sample.wav"]

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = BUNDLE.get_model().eval()
    wrapper = AcousticExport(model, args.hidden_layer).eval()
    n_layers = len(model.encoder.transformer.layers)
    print(f"wav2vec2-base: {n_layers} layers, emitting logits + layer {args.hidden_layer}")

    # 1. The wrapper must be the pipeline, not merely close to it. Checked
    #    before export so a mistake here is never baked into a graph.
    print("\nwrapper against the pipeline")
    for clip in clips:
        wav = read_wav(clip)
        ref_logits, _ = model(wav)
        ref_hidden = model.extract_features(wav)[0][args.hidden_layer - 1]
        got_logits, got_hidden = wrapper(wav)
        d_log = float((got_logits - ref_logits).abs().max())
        d_hid = float((got_hidden - ref_hidden).abs().max())
        print(f"  {clip.name:<24} logits {d_log:.3e}  layer {args.hidden_layer} {d_hid:.3e}")
        if d_log != 0.0 or d_hid != 0.0:
            raise SystemExit("wrapper diverges from the pipeline; refusing to export")

    # 2. Export. Dynamic on samples and frames: clip length is a request
    #    parameter, and a fixed axis would silently reshape short audio.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".onnx.tmp")
    print(f"\nexporting -> {args.out}")
    torch.onnx.export(
        wrapper,
        (read_wav(clips[0]),),
        str(tmp),
        input_names=["waveform"],
        output_names=["logits", "hidden"],
        dynamic_axes={
            "waveform": {0: "batch", 1: "samples"},
            "logits": {0: "batch", 1: "frames"},
            "hidden": {0: "batch", 1: "frames"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # 3. Parity through onnxruntime, on audio the export did not see, at more
    #    than one length -- a graph that froze a dimension passes at the traced
    #    length and fails everywhere else.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    print("onnxruntime against the pipeline")
    worst = {"logits": 0.0, "hidden": 0.0}
    worst_rel = {"logits": 0.0, "hidden": 0.0}
    for clip in clips:
        wav = read_wav(clip)
        for frac, tag in ((1.0, "full"), (0.5, "half")):
            w = wav[:, : max(int(wav.shape[1] * frac), BUNDLE.sample_rate // 4)]
            ref_logits, _ = model(w)
            ref_hidden = model.extract_features(w)[0][args.hidden_layer - 1]
            got = sess.run(None, {"waveform": w.numpy()})
            rl, rh = ref_logits.numpy(), ref_hidden.numpy()
            d_log = float(np.abs(got[0] - rl).max())
            d_hid = float(np.abs(got[1] - rh).max())
            r_log = d_log / float(np.abs(rl).max())
            r_hid = d_hid / float(np.abs(rh).max())
            worst["logits"] = max(worst["logits"], d_log)
            worst["hidden"] = max(worst["hidden"], d_hid)
            worst_rel["logits"] = max(worst_rel["logits"], r_log)
            worst_rel["hidden"] = max(worst_rel["hidden"], r_hid)
            # The decode and the alignment read the argmax, not the value.
            if not bool((got[0].argmax(-1) == rl.argmax(-1)).all()):
                raise SystemExit("argmax disagrees with the pipeline; refusing to install")
            shape_ok = got[0].shape == tuple(ref_logits.shape) and got[1].shape == tuple(ref_hidden.shape)
            print(f"  {clip.name:<20} {tag:<5} logits {d_log:.3e} ({r_log:.1e} rel)"
                  f"  hidden {d_hid:.3e} ({r_hid:.1e} rel)"
                  f"  shapes {'ok' if shape_ok else 'MISMATCH'}")
            if not shape_ok:
                raise SystemExit("shape mismatch; refusing to install")
    if max(worst_rel.values()) > args.tolerance:
        raise SystemExit(f"parity {max(worst_rel.values()):.3e} over tolerance {args.tolerance:.1e}")

    tmp.replace(args.out)
    digest = sha256(args.out)
    size = args.out.stat().st_size
    check = {
        "hidden_layer": args.hidden_layer,
        "layers": n_layers,
        "opset": args.opset,
        "max_abs": worst,
        "max_rel": worst_rel,
        "tolerance_rel": args.tolerance,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "bytes": size,
        "sha256": digest,
    }
    (args.out.parent / "export_check.json").write_text(json.dumps(check, indent=2) + "\n")
    print(f"\n{size} bytes  sha256 {digest}")
    print(f"worst logits {worst['logits']:.3e} ({worst_rel['logits']:.1e} rel)"
          f"  worst hidden {worst['hidden']:.3e} ({worst_rel['hidden']:.1e} rel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
