# Models

## Acoustic

Exported from `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H` (character CTC, 16 kHz). Third-party weights; see the Facebook / torchaudio model cards. This repo's TypeScript is Apache-2.0 and does not re-license that checkpoint.

It is an ASR model, not a pronunciation-assessment model. Everything downstream — alignment, GOP, edits — is derived from its per-frame posteriors.

## Scoring heads

None ship yet, so `scores.pronunciation`, `scores.fluency` and `scores.overall` are `null`.

`scores.content` is always present: it comes from counting edits, not from a model.

A scoring backend plugs in through the `ScoringBackend` interface in `@readaloudkit/types`. It receives the word list, the per-word GOP features, the 21 utterance prosody features and the content breakdown.

## Feature contracts

A head is trained against a fixed feature order:

- `WORD_FEATURE_KEYS` (`@readaloudkit/gop`) — 11 per-word acoustic features
- `UTTERANCE_FEATURE_KEYS` (`@readaloudkit/features`) — 21 prosody features
- `UTTERANCE_GOP_KEYS` (`@readaloudkit/gop`) — 9 aggregates appended after those

Feature extraction for training runs through these same packages, so there is no second implementation that can drift from the one used at serving time.
