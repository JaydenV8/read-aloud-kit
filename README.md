# ReadAloudKit

[![CI](https://github.com/JaydenV8/read-aloud-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/JaydenV8/read-aloud-kit/actions/workflows/ci.yml)

Read-aloud pronunciation and fluency scoring in TypeScript. Reference text plus
audio in; a word-level error map, per-word acoustic evidence, and utterance
scores out — from one wav2vec2 forward pass, on CPU, in Node.

**Not affiliated with or endorsed by Pearson.** Not an official PTE scorer, and
not a reimplementation of Pearson's scoring system. The 10–90 range it reports is
a presentation choice calibrated against nothing. See [Limitations](#limitations).

```bash
pnpm install && pnpm models:download && pnpm dev
```

```bash
curl -X POST http://127.0.0.1:3000/v1/read-aloud/analyze \
  -F "audio=@examples/sample.wav" \
  -F "referenceText=The university provides many opportunities."
```

```jsonc
{
  "hypothesis": "the university provides many opportunities",
  "words": [ /* status, timings, GOP, band, needsAttention — one per word */ ],
  "content": { "score": 5, "maxScore": 5, "strict": 5, "calibrated": 5 },
  "prosody": { "wpm": 146.5, "coverage": 1, /* 21 features in total */ },
  "pauses": [ { "startMs": 2449, "endMs": 2570, "durMs": 121, "where": "tail" } ],
  "tips":   [ /* rule-derived, from the error list */ ],
  "scores": { "backend": "community", "content": 5,
              "pronunciation": 73.7, "fluency": 81.8, "overall": 76.5 }
}
```

The scoring heads are checked into `releases/` — about a megabyte — so a clone
scores without a download step. Only the 378 MB acoustic model is fetched.

## How it works

```text
reference text ───────────────────────────┐
                                          │
audio → wav2vec2 CTC ──┬─ forced align → per-word GOP ─┐
   (one ONNX forward)  │                               ├─→ word bands
                       └─ greedy decode → hypothesis ──┤    pronunciation
                                          │            │    fluency
                                          └─ word edit alignment
                                                       ↓
                                      omission · substitution · insertion
                                      content · prosody · pauses · tips
```

Both channels come off the same forward pass, and they are kept separate on
purpose. A transcript comparison cannot see a single skipped short word — the
decoder simply produces a fluent-looking transcript without it. The forced
aligner has no such freedom: it must place every reference word somewhere, so a
skipped word gets squeezed onto silence and its mean posterior falls by an order
of magnitude. `words[].statusEvidence` records which channel decided the call.

Feature extraction for training runs through the same TypeScript packages that
serve requests, so there is no second implementation to drift against. The
alignment and GOP code is additionally pinned frame-by-frame to
`torchaudio.functional.forced_align` goldens in `tests/fixtures/`.

## Results

`0.5-community`, on the corpus's official test split — 2500 utterances, 15967
words, from speakers no training or selection step touched.

Word bands. `good` covers 88.1% of words, so **answering `good` every time scores
0.881** and accuracy alone is a poor summary:

| | precision | recall | F1 | support |
|---|---|---|---|---|
| good | 0.936 | 0.918 | 0.927 | 14062 |
| average | 0.281 | 0.241 | 0.259 | 1010 |
| bad | 0.351 | 0.514 | 0.417 | 895 |

Accuracy 0.853, macro F1 0.535.

The decision that actually ships is binary — is this word worth pointing at —
and it runs **precision 0.472, recall 0.539**, flagging 13.6% of words against a
true rate of 11.9%. Nearly one flag in two is real and about half the real
problems get flagged: a usable hint, and a bad verdict. Separating `average`
from `bad` is weaker still, so the API exposes `needsAttention` as the field to
render and documents `level` as secondary.

Utterance scores, Pearson r against the expert label:

| head | r | MAE | MAE predicting the mean |
|---|---|---|---|
| accuracy → `scores.pronunciation` | 0.665 | 0.849 | 1.131 |
| fluency → `scores.fluency` | 0.737 | 0.693 | 1.056 |
| total → `scores.overall` | 0.687 | 0.825 | 1.182 |

GOPT (ICASSP 2022) reports 0.742 sentence-level PCC on this corpus with a
transformer over per-phone GOP vectors. These are gradient-boosted trees over 32
utterance features and land below it — the expected trade for 1.0 MB of weights
that run on CPU in Node.

### What moved the numbers

`0.4-community` → `0.5-community` is a controlled change: same backbone, same
alignment, same audio, same split, same decision rule. The only difference is
what a word is summarised as — eleven averages, versus 36 features that keep the
shape of the per-character posterior and duration series plus one word of
context either side.

| | 0.4-community | 0.5-community |
|---|---|---|
| word macro F1 | 0.511 | **0.535** |
| `bad` F1 | 0.357 | **0.417** |
| flag precision | 0.417 | **0.472** |
| flag recall | 0.562 | 0.539 |
| words flagged | 2564 | **2174** |
| pronunciation r | 0.652 | **0.665** |
| fluency r | 0.729 | **0.737** |
| overall r | 0.684 | **0.687** |

The gain lands where it is worth having: the flag gets more precise while
flagging 390 *fewer* words, and `bad` — rarest and most consequential — improves
most. The utterance heads barely move, which is what a six-word corpus predicts:
those features describe how delivery is distributed across an utterance, and six
words leave little to distribute.

Both generations stay in `releases/`, so the comparison is re-runnable rather
than asserted:

```bash
pnpm heads:eval --onnx releases/0.4-community --split test
pnpm heads:eval --onnx releases/0.5-community --split test
```

## Evaluation discipline

The numbers above are only worth what the protocol behind them is worth.

- **The official speaker-disjoint split is used as-is** and never re-split.
  Validation is 25 speakers held out of the *training* half, so early stopping
  never sees a test speaker. `train.py` asserts the disjointness at runtime
  rather than trusting it.
- **Fitted constants are derived here, not inherited.** `WEAK_CHAR_GOP` is swept
  on the training split — minus the speakers used for early stopping, so it is
  not selected against the same data twice — and lands at −4.3. The objective is
  flat between about −5 and −2.5, so the value matters less than the fact that it
  was measured on this corpus instead of imported from another.
- **Features that would cheat are excluded, and the cost is reported.** Every
  speaker was recorded once, so recording conditions track the speaker and the
  speaker tracks proficiency; a head leaning on gain would look good for the
  wrong reason. Dropping all four signal-level features moved test correlation by
  at most 0.008. The prompt-length-correlated features are dropped for the same
  class of reason: they are the *strongest* predictors on a corpus of six-word
  utterances and would be dominated by prompt length at the 40–80 words this is
  aimed at.
- **Export is checked, not assumed.** ONNX conversion is compared against the
  in-process model and fails the build on drift — the shipped classifier lands
  1.2e-07 against a 2e-04 tolerance. `pnpm runtime:verify` then pushes corpus
  audio through the entire Node serving path and checks it reaches the same
  answers.
- **Raw metrics and shipped metrics are labelled separately.** A manifest records
  raw argmax; the model card reports the thresholded decision the runtime
  actually makes. They differ, and conflating them silently flatters the model.
- **Negative results are kept.** The stress head reaches 0.044 precision and the
  per-character features raise it to 0.053 — right direction, nowhere near
  usable — so it is trained, exported for reproducibility, and deliberately not
  loaded. `MODEL_CARD.md` says why.

Every run records corpus SHA-256, split, seed, feature version, fitted constants
and toolchain versions into the release manifest.

## Limitations

Read `MODEL_CARD.md` before putting any of this in front of a learner.

- **Domain shift is unmeasured.** The corpus is 1–10 words, median 6; the target
  is 40–80. The feature choices above are a design argument for transferring, not
  evidence of it. Nothing here has been evaluated at target length.
- **The 10–90 range is presentation, not calibration.** It is a linear rescale of
  a 0–10 corpus score, fitted to nothing, and it is not an examination score.
- **Speaker population is narrow.** All Mandarin-native, roughly half children.
  Other first languages are untested.
- **Word bands are hints.** 0.472 precision on the binary flag; weaker on the
  three-way band. Render them as "worth listening to again", never as a verdict.
- **Reading only.** A known reference text is assumed throughout. There is no
  free-speech mode.

## Reproducing

```bash
pnpm corpus:download          # speechocean762, SHA-256 pinned in corpus.lock.json
pnpm corpus:prepare
pnpm features:extract         # through the serving code path
python training/sweep_weak_char.py
pnpm features:extract         # again, once the constant is set
pnpm heads:train
pnpm heads:export             # ONNX + parity check
pnpm heads:eval               # the numbers above
pnpm runtime:verify           # the same heads through the Node path
```

Feature extraction is deterministic: re-running it from the pinned corpus
reproduces `features.jsonl` byte for byte.

## Layout

| | |
|---|---|
| `packages/audio` | WAV decode, mono, 16 kHz resample; ffmpeg only for other containers |
| `packages/inference` `packages/ctc` | ONNX Runtime session, greedy decode |
| `packages/alignment` `packages/gop` | Forced alignment, per-word and per-character GOP |
| `packages/edits` | Reference↔hypothesis alignment, content scoring |
| `packages/features` `packages/prosody` | Utterance features, adaptive pause detection |
| `packages/scoring` | Head loading, feature-contract assembly, band decision |
| `packages/core` | The analyzer that ties them together |
| `apps/api` `packages/sdk` `packages/cli` | Hono service, typed client, CLI |
| `training/` | Python: corpus prep, LightGBM, ONNX export, evaluation |
| `examples/` | `sample.wav` — synthetic speech, reproducible; see `examples/README.md` |
| `releases/` | Weights and manifests, one directory per generation |

```bash
pnpm test      # 60 tests, including the cross-language goldens
pnpm oxlint
```

The suite needs no acoustic model — alignment and GOP run against the
torchaudio goldens, scoring against the checked-in heads — so CI is
`pnpm install && pnpm test` and finishes in seconds.

## License

Apache-2.0 for the source (`LICENSE`). The scoring heads in `releases/` are
CC BY 4.0, matching the corpus they derive from — see `LICENSE-MODELS` for the
attribution to reproduce and `NOTICE` for the citation. The acoustic checkpoint
is third-party under its own license; see `docs/models.md`.

Package version (`0.1.0`) and head version (`0.5-community`) are separate
sequences: the heads can be retrained without the API changing, and the reverse.
