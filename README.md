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
              "pronunciation": 75.8, "fluency": 78.0, "overall": 72.6 }
}
```

The scoring heads are checked into `releases/` — about a megabyte — so a clone
scores without a download step. Only the 378 MB acoustic model is fetched.

The graph emits two tensors from one pass: the CTC logits everything downstream
is derived from, and one intermediate layer. The second is there because a logit
layer is trained to discard whatever does not separate one character from the
next, and measured against expert labels that is most of what a fluency score
wants. See [What moved the numbers](#what-moved-the-numbers).

## How it works

```text
reference text ───────────────────────────┐
                                          │
audio → wav2vec2 ──┬── layer 3 ──── pooled per word ────┐
  (one ONNX        │                                    │
   forward)        └── CTC logits ─┬─ forced align → GOP ┼─→ word bands
                                   │                    │   pronunciation
                                   └─ greedy decode ────┤   fluency
                                          │             │
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

`0.6-community`, on the corpus's official test split — 2500 utterances, 15967
words, from speakers no training or selection step touched.

> 0.6 reads a tensor the earlier acoustic asset did not emit, so
> `pnpm models:download` fetches a newer graph (tag `acoustic-v2`) whose
> `logits` are bit-identical to the previous one. Older checkouts pin the older
> asset and are unaffected.

Word bands. `good` covers 88.1% of words, so **answering `good` every time scores
0.881** and accuracy alone is a poor summary:

| | precision | recall | F1 | support |
|---|---|---|---|---|
| good | 0.933 | 0.927 | 0.930 | 14062 |
| average | 0.307 | 0.258 | 0.281 | 1010 |
| bad | 0.382 | 0.491 | 0.429 | 895 |

Accuracy 0.860, macro F1 0.547.

The decision that actually ships is binary — is this word worth pointing at —
and it runs **precision 0.487, recall 0.511**, flagging 12.5% of words against a
true rate of 11.9%. Nearly one flag in two is real and about half the real
problems get flagged: a usable hint, and a bad verdict. Separating `average`
from `bad` is weaker still, so the API exposes `needsAttention` as the field to
render and documents `level` as secondary.

Utterance scores, Pearson r against the expert label:

| head | r | MAE | MAE predicting the mean |
|---|---|---|---|
| accuracy → `scores.pronunciation` | 0.703 | 0.798 | 1.131 |
| fluency → `scores.fluency` | 0.761 | 0.662 | 1.056 |
| total → `scores.overall` | 0.732 | 0.760 | 1.182 |

Measured again through the Node serving path on the same 2500 utterances, these
reproduce to three decimals (0.7029 / 0.7610 / 0.7320), so they are what a
request returns rather than what an evaluation script computes.

GOPT (ICASSP 2022) reports 0.742 sentence-level PCC on this corpus with a
transformer over per-phone GOP vectors. `scores.overall` is at 0.732 from
gradient-boosted trees over 64 features, in 1.1 MB of weights on CPU in Node.

### What moved the numbers

Three generations, each a controlled change on the same backbone, alignment,
audio, split and decision rule. 0.5 changed what a *word* is summarised as —
eleven averages versus 36 features keeping the shape of the per-character series.
0.6 changed which *layer* of the same forward pass is read.

| | 0.4-community | 0.5-community | 0.6-community |
|---|---|---|---|
| word macro F1 | 0.511 | 0.535 | **0.547** |
| `bad` F1 | 0.357 | 0.417 | **0.429** |
| flag precision | 0.417 | 0.472 | **0.487** |
| flag recall | 0.562 | 0.539 | 0.511 |
| words flagged | 2564 | 2174 | **2000** |
| pronunciation r | 0.652 | 0.665 | **0.703** |
| fluency r | 0.729 | 0.737 | **0.761** |
| overall r | 0.684 | 0.687 | **0.732** |

0.5's gain was at word level; its utterance heads barely moved, which is what a
six-word corpus predicts of features describing how delivery is distributed
across an utterance. 0.6 lands the other way round, and larger: the utterance
heads gain three to twelve times what the entire per-character rework gave them.

The reason is that the CTC logits are the *end* of a pipeline trained to keep
only what separates one character from the next. **Every one of wav2vec2's twelve
layers is more useful to these heads than the last one** — and the last one is
what a CTC export normally emits. Reading layer 3 as well costs one extra output
tensor, 1.4% latency, and no extra arithmetic: the forward pass already computed
it.

Layer 3 was chosen by 5-fold speaker-disjoint cross-validation over the training
half, refitting the projection inside every fold; the word head and the utterance
heads picked it independently. A single 25-speaker holdout had picked layer 9,
which cross-validation shows is genuinely worse — the holdout was not imprecise,
it was wrong.

Pooling was swept and did not matter. Mean, standard deviation, first and last
frame, thirds, extremes, all at a fixed feature budget: nothing beat the plain
mean on either head. A word here is about six frames, which is very little series
to have a shape — a negative result that should be revisited at the 40–80 word
prompts this is aimed at, not treated as settled.

All three generations stay in `releases/`, so the comparison is re-runnable
rather than asserted:

```bash
pnpm heads:eval --onnx releases/0.4-community --split test
pnpm heads:eval --onnx releases/0.5-community --split test
pnpm heads:eval --onnx releases/0.6-community --features training/data/features_v3.jsonl
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

Package version (`0.1.0`), head version (`releases/CURRENT`, today
`0.6-community`) and acoustic asset tag (`acoustic-v2`) are separate sequences:
the heads can be retrained without the API changing, and the reverse.
