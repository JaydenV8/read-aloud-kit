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

`0.5-community` is fitted on:

- `WORD_FEATURE_KEYS_V2` (`@readaloudkit/gop`) — 36 per-word features. Eleven of
  them are the v1 summary; the rest keep the shape of the per-character posterior
  and duration series, plus one word of context either side
- `UTTERANCE_FEATURE_KEYS` (`@readaloudkit/features`) — 21 prosody features
- `UTTERANCE_GOP_KEYS_V2` (`@readaloudkit/gop`) — 24 aggregates appended after
  those, giving 45; the shipped heads read 32 of them

`0.4-community` is fitted on the v1 lists — `WORD_FEATURE_KEYS` (11) and
`UTTERANCE_GOP_KEYS` (9), 19 of 30 utterance features — and still loads, because
the v2 lists are supersets whose shared keys carry identical values.

Feature extraction for training runs through these same packages, so there is no second implementation that can drift from the one used at serving time.
