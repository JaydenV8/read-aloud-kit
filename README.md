# ReadAloudKit

Open-source toolkit for read-aloud speech analysis, CTC decoding, forced alignment, pronunciation features, fluency features, and word-level error detection.

Designed for PTE-style Read Aloud practice and speech-assessment research.

**ReadAloudKit is an independent project and is not affiliated with or endorsed by Pearson.** It is not an official PTE scorer and does not reproduce Pearson’s proprietary scoring system.

## Features

- WAV decode, mono, 16 kHz resample (ffmpeg optional for other formats)
- wav2vec2-base CTC inference via ONNX Runtime (Node.js)
- One forward pass → forced alignment / GOP **and** greedy decode
- Word-level omission / substitution / insertion / repetition, with per-word timings
- Per-word acoustic evidence (GOP, posterior margin, blank ratio, rate)
- Content score, strict and calibrated
- Utterance prosody: articulation rate, coverage, pause statistics, energy
- Adaptive pause detection that scales to the speaker
- Rule-driven tips built from the error list
- Hono HTTP API and TypeScript SDK
- Pluggable scoring backend for pronunciation / fluency (none shipped yet)

## Quick start

Primary runtime is **Node.js**.

```bash
pnpm install
pnpm models:download
pnpm dev
```

```bash
curl -X POST \
  http://127.0.0.1:3000/v1/read-aloud/analyze \
  -F "audio=@examples/sample.wav" \
  -F "referenceText=The university provides many opportunities."
```

Without a Community scoring checkpoint, `scores.pronunciation` / `scores.fluency` / `scores.overall` are `null`. Everything else — hypothesis, word timings, edits, content, prosody, pauses, tips — is returned either way.

```bash
pnpm analyze -- analyze --audio examples/sample.wav --text "The university provides many opportunities."
```

## Architecture

```text
Reference Text ─────────────────────────┐
                                       │
Audio                                  │
  ↓                                    │
wav2vec2 CTC  (single ONNX forward)    │
  ↓                                    │
  ├─ forced alignment / GOP ───────────┤
  └─ greedy CTC decode                 │
             ↓                         │
         hypothesis                    │
             ↓                         │
       word alignment ←────────────────┘
             ↓
 omission / substitution / insertion / repetition
             ↓
 content · prosody · pauses · tips
             ↓
 optional scoring backend (not included)
```

The edit alignment and the acoustics are kept as separate channels: a single
skipped short word is invisible to a transcript comparison but obvious in the
forced alignment, which has to park the word on silence somewhere. See
`docs/architecture.md`.

## What ships

| | Status |
|---|---|
| Acoustic CTC | Facebook wav2vec2-base-960h, fetched by `pnpm models:download` |
| Word-level analysis, content, prosody, pauses, tips | included, no extra model needed |
| Pronunciation / fluency heads | community heads trained on speechocean762, 1.0 MB |

The scoring heads are optional: without them the analysis still runs and the
model-derived fields are `null`. Read `MODEL_CARD.md` before displaying any of
their output — the word flag is a hint at 0.42 precision, and the 10–90 range is
a presentation choice, not an examination score.

## Tooling

Vite+ (`vp`) plus Oxlint and Oxfmt:

```bash
pnpm test
pnpm oxlint
pnpm oxfmt
```

## License

Apache-2.0 for this repository’s source. The optional acoustic checkpoint is a third-party pretrained model — see `docs/models.md`.
