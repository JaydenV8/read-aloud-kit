# Models

## Acoustic

Exported from `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H` (character CTC, 16 kHz). Third-party weights; see the Facebook / torchaudio model cards. This repo's TypeScript is Apache-2.0 and does not re-license that checkpoint.

It is an ASR model, not a pronunciation-assessment model. Everything downstream — alignment, GOP, edits — is derived from its per-frame posteriors.

## Scoring heads

The community heads are LightGBM models exported to ONNX, 1.3 MB for the four the
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

Install by placing the exported files in `models/scoring/` next to
`scoring.json`, or point `READALOUDKIT_SCORING` at another directory. Without
them the analyzer falls back to the noop backend and the model-derived fields
stay `null`.

A different backend plugs in through the `ScoringBackend` interface in
`@readaloudkit/types`. It receives the word list, the per-word GOP features, the
21 utterance prosody features and the content breakdown.

## Feature contracts

A head is trained against a fixed feature order:

- `WORD_FEATURE_KEYS` (`@readaloudkit/gop`) — 11 per-word acoustic features
- `UTTERANCE_FEATURE_KEYS` (`@readaloudkit/features`) — 21 prosody features
- `UTTERANCE_GOP_KEYS` (`@readaloudkit/gop`) — 9 aggregates appended after those

Feature extraction for training runs through these same packages, so there is no second implementation that can drift from the one used at serving time.
