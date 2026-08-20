# Models

The acoustic model lives here and is **not** stored in git — at 378 MB it is
fetched. The scoring heads are a different matter: they are about a megabyte and
are checked into `releases/`, so they need no download. `models/scoring/` is
their override slot, which is why it is gitignored. See `docs/models.md`.

| File | Class | Role |
|---|---|---|
| `wav2vec2-base-960h-ctc.onnx` | EXTERNAL_PRETRAINED | Acoustic CTC (Facebook wav2vec2-base-960h via torchaudio) |
| `labels.json` | EXTERNAL_PRETRAINED | CTC vocabulary, blank=0, 16 kHz |

The graph emits `logits` and, from the same pass, one intermediate transformer
layer as `hidden`. Read them by name: a hidden layer has the same rank as the
logits and differs only in its last dimension, so a positional read yields a
plausible tensor and nonsense downstream. `training/export_acoustic.py`
(`pnpm models:export`) rebuilds it and refuses to write a file that disagrees
with the PyTorch pipeline.

```bash
pnpm models:download
```

Resolution order: an already-installed valid copy → a local export (`READALOUDKIT_MODEL_SRC`, `/tmp/rak-models/`) → the GitHub release asset (378 MB, tag `acoustic-v2`). The download is checked against a pinned SHA-256 and a mismatched file is deleted rather than used. Override the source with `READALOUDKIT_MODEL_URL`.

Without this file nothing runs — it is the one forward pass everything downstream is derived from. `scores.content` is the exception: it comes from a rule and needs no model at all.

## Provenance and license

The acoustic graph is an ONNX export of `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`, i.e. Facebook's `wav2vec2-base-960h`, pretrained and fine-tuned on LibriSpeech 960h. Those weights are third-party and are redistributed here unmodified except for the ONNX graph conversion; they are covered by their original upstream license, not by this repository's Apache-2.0. See the [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) model card and the torchaudio pipeline documentation.

The export was checked against the PyTorch pipeline frame by frame; `export_check.json` records the maximum absolute logit difference.

| File | SHA-256 |
|---|---|
| `wav2vec2-base-960h-ctc.onnx` | `a5162c4510b81c13d7c5dab8c80c577caf643bfcada663d74a6fcc9c02b57356` |
