# Models

## Acoustic

Exported from `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H` (character CTC, 16 kHz). Third-party weights; see the Facebook / torchaudio model cards. This repo's TypeScript is Apache-2.0 and does not re-license that checkpoint.

It is an ASR model, not a pronunciation-assessment model. Everything downstream — alignment, GOP, edits — is derived from its per-frame posteriors.

## Scoring heads

The community heads are LightGBM models exported to ONNX, 1.0 MB for the four the
runtime loads. They are trained on speechocean762 (CC BY 4.0) and nothing else —
see `MODEL_CARD.md` for data, splits, held-out results and limitations, and
`NOTICE` for the attribution that travels with them.

| output | head |
|---|---|
| `words[].level`, `words[].needsAttention` | `word_level` |
| `scores.pronunciation` | `utterance_accuracy` |
| `scores.fluency` | `utterance_fluency` |
| `scores.overall` | `utterance_total` |

`word_stress` and `utterance_prosodic` are exported for reproducibility but not
loaded. Stress reaches 0.044 precision on held-out data, which is worse than
saying nothing.

`scores.content` is always present: it comes from counting edits, not a model.

## Where they load from

Checked in, so a clone scores without a download. Resolution order:

1. `READALOUDKIT_SCORING`, if set
2. `models/scoring/`, the override slot for a local export — gitignored
3. the directory named in `releases/CURRENT`

`GET /health` reports which of these won, as `scoring: { backend, version }`.
Remove them all and the analyzer falls back to the noop backend: the analysis
still runs and the model-derived fields are `null`.

`releases/` keeps every generation, not just the current one, so the comparison
in `MODEL_CARD.md` can be re-run rather than taken on trust:

```bash
training/.venv/bin/python training/eval.py --onnx releases/0.4-community --split test
training/.venv/bin/python training/eval.py --onnx releases/0.5-community --split test
```

A different backend plugs in through the `ScoringBackend` interface in
`@readaloudkit/types`. It receives the word list, the per-word GOP features, the
utterance prosody features and the content breakdown.

## Feature contracts

A head is trained against a fixed feature order, and the release carries the
order it was fitted on. The runtime assembles its input from that shipped key
list — `manifest.wordFeatureKeys` and `manifest.utteranceFeatureKeys` — rather
than from a version number, so an old release keeps working after the packages
grow new features. Loading a head whose ONNX input width disagrees with its
manifest is refused rather than served.

`0.6-community` is fitted on everything `0.5-community` uses, plus a projection
of an intermediate acoustic layer:

- `WORD_FEATURE_KEYS_V2` (`@readaloudkit/gop`) — 36 per-word features. Eleven of
  them are the v1 summary; the rest keep the shape of the per-character posterior
  and duration series, plus one word of context either side
- `UTTERANCE_FEATURE_KEYS` (`@readaloudkit/features`) — 21 prosody features
- `UTTERANCE_GOP_KEYS_V2` (`@readaloudkit/gop`) — 24 aggregates appended after
  those, giving 45; the shipped heads read 32 of them
- `hid0`..`hid31` — 32 components of transformer layer 3, mean-pooled over the
  word span for the word head and over the whole clip for the utterance heads,
  appended as a contiguous tail to both key lists

`0.5-community` is the same without the `hid*` tail, and `0.4-community` is
fitted on the v1 lists — `WORD_FEATURE_KEYS` (11) and `UTTERANCE_GOP_KEYS` (9),
19 of 30 utterance features. Both still load, because each later list is a
superset whose shared keys carry identical values.

## The hidden projection

768 columns of pooled transformer output would let a head fitted on 2000
utterances memorise speakers, so a release ships its own reduction of them:
`hidden_word.onnx` and `hidden_utterance.onnx`, about 96 KB each.

The projection belongs to the release rather than to the acoustic model. It is
fitted on this corpus's training split, while the checkpoint stays a stock
third-party export that happens to expose one more tensor — which is what keeps
the checkpoint reusable and the fitted part attributable.

Standardisation and PCA are both affine, so they fold into a single `y = xA + b`
and ship as one `Gemm`. The fold is checked against running the two in sequence
before the graph is written, because it is what every request runs while the
sequential form is only ever run at fit time.

Words and whole clips get separate fits: they are pooled over different spans
and their covariance is not the same.

A release that declares `hiddenProjection` cannot be served by an acoustic model
that emits only logits. The backend refuses rather than filling zeros, since
zeros would return numbers that look like scores and are not. The reverse — a
graph with a hidden layer serving a release that ignores it — is fine, and is
what `0.4`/`0.5` do.

Feature extraction for training runs through these same packages, so there is no second implementation that can drift from the one used at serving time.
