# Model card — ReadAloudKit community heads

Pronunciation and fluency heads for read-aloud practice. They turn the acoustic
features in `@readaloudkit/gop` and `@readaloudkit/features` into a per-word
band and three utterance scores.

**These are not PTE scores.** They are not produced by, endorsed by, or
comparable to Pearson's scoring system. See [Scale](#scale).

| | |
|---|---|
| Version | 0.6-community |
| Model | LightGBM, exported to ONNX |
| Size | 1.1 MB of heads plus 0.2 MB of projection |
| Training data | speechocean762 only |
| Word features | 36 (`WORD_FEATURE_KEYS_V2`) + 32 hidden |
| Utterance features | 32 of 45, + 32 hidden |
| Acoustic layer read | transformer layer 3 **and** the CTC logits |
| Runtime | onnxruntime-node, CPU |

Releases are numbered to match the generation they correspond to, with a
`-community` suffix. `0.4-community` summarised each word as eleven averages;
`0.5-community` keeps the shape of its per-character series; `0.6-community`
adds a projection of an intermediate acoustic layer. All three are in
`releases/`, and a release carries the feature contract it was fitted on, so the
runtime assembles its input from the shipped key list rather than from a version
number.

**`0.6-community` needs an acoustic model that emits a hidden layer.** Served
against a graph that only emits logits it refuses to score rather than filling
zeros, because zeros would return numbers that look like scores and are not.

> `pnpm models:download` fetches that graph: release tag `acoustic-v2`, whose
> `logits` are bit-identical to the `v0.1.0` asset it supersedes. An older
> checkout pins the older asset and keeps working — see
> [How 0.6 was promoted](#how-06-was-promoted).

## Training data

[speechocean762](https://www.openslr.org/101/), CC BY 4.0. 5000 English
utterances read by Mandarin-native learners aged 6–43, scored independently by
five experts at the phoneme, word and utterance level.

> speechocean762: An Open-Source Non-Native English Speech Corpus for
> Pronunciation Assessment. Interspeech 2021.

No other corpus contributed labels, validation, model selection or any fitted
constant. `training/artifacts/provenance.json` records the corpus checksum, the
split, the seed and the toolchain versions for each run.

## Splits

The corpus ships a speaker-disjoint 2500/2500 split and it is used as-is.
Validation comes from 25 speakers held out of the official training split, so
early stopping never sees the test speakers. Every number below is on the
official test split, which no training or selection step touched.

## Results

### Word band

15967 words. `good` covers 88.1% of them, so accuracy alone is a poor summary —
answering `good` every time scores 0.881.

| | precision | recall | F1 | support |
|---|---|---|---|---|
| good | 0.933 | 0.927 | 0.930 | 14062 |
| average | 0.307 | 0.258 | 0.281 | 1010 |
| bad | 0.382 | 0.491 | 0.429 | 895 |

Accuracy 0.860, macro F1 0.547.

The single decision that matters — is this word worth pointing at — runs
**precision 0.487, recall 0.511**, flagging 12.5% of words against a true rate of
11.9%.

So: nearly one flag in two is real, and about half of the real problems get
flagged. That is a usable hint and a bad verdict. A display should treat a flag
as "worth listening to again", never as "you said this wrong". Telling `average`
from `bad` is still weak — `average` sits near 0.28 precision — so consider
rendering one highlight rather than two shades.

### Across the three generations

Same corpus, same official split, same decision rule, same harness, same audio.
0.4 → 0.5 changes only what a *word* is summarised as. 0.5 → 0.6 changes only
which *layer* of the same forward pass is read.

| | 0.4-community | 0.5-community | 0.6-community |
|---|---|---|---|
| word accuracy | 0.831 | 0.853 | **0.860** |
| word macro F1 | 0.511 | 0.535 | **0.547** |
| `bad` F1 | 0.357 | 0.417 | **0.429** |
| flag precision | 0.417 | 0.472 | **0.487** |
| flag recall | 0.562 | 0.539 | 0.511 |
| words flagged | 2564 | 2174 | **2000** |
| accuracy r | 0.652 | 0.665 | **0.703** |
| fluency r | 0.729 | 0.737 | **0.761** |
| total r | 0.684 | 0.687 | **0.732** |

0.5's gain was concentrated at word level: the flag got more precise while
flagging 390 fewer words, and the utterance heads barely moved — which is what a
six-word corpus predicts of features describing how delivery is distributed
across an utterance.

0.6 lands the other way round. The word head improves modestly and its recall
falls again, but the utterance heads gain three to twelve times what the whole
per-character rework gave them. That is the difference between summarising the
same representation better and reading a different representation: the last
transformer layer is trained to keep only what separates one character from the
next, and most of what an utterance score wants was discarded before it.

**Every one of the twelve layers is more useful to these heads than the last
one, and the last one is what a CTC export normally emits.** Layer 3 was chosen
by 5-fold speaker-disjoint cross-validation over the training half, refitting the
projection inside each fold; the word head and the utterance heads picked it
independently. Layers 2 and 4 are indistinguishable from it and layers 5–12 are
all significantly worse for the word head, so the choice is narrow rather than
arbitrary.

Pooling was swept too and did not matter: mean, standard deviation, first and
last frame, thirds and extremes were compared at a fixed feature budget and
nothing beat the plain mean on either head. A word here is about six frames,
which is very little series to have a shape.

### Utterance scores

2500 utterances, Pearson correlation against the expert score.

| head | r | MAE | MAE if you just predicted the mean |
|---|---|---|---|
| accuracy → `scores.pronunciation` | 0.703 | 0.798 | 1.131 |
| fluency → `scores.fluency` | 0.761 | 0.662 | 1.056 |
| total → `scores.overall` | 0.732 | 0.760 | 1.182 |

Reproduced through the Node serving path on the same 2500 utterances —
0.7029 / 0.7610 / 0.7320, matching the offline figures to three decimals — so
these are what a request returns and not only what an evaluation script
computes. `training/verify_scores.ts` is that check.

For context, GOPT (ICASSP 2022) reports 0.742 sentence-level PCC on this corpus
with a transformer over per-phone GOP vectors. `scores.overall` now lands at
0.732 from gradient-boosted trees over 64 features, at 1.1 MB of weights on CPU
in Node.

## Scale

The corpus scores 0–10. The API reports 10–90 because that is the range readers
of a read-aloud report expect. The mapping is linear and fitted to nothing:

```
display = 10 + (raw / 10) * 80
```

It does not convert a corpus score into an official score. Nothing here is
calibrated against any official examination.

`scores.content` does not come from a model at all — it counts edits against the
reference text, and is on 0–5.

## What is deliberately not shipped

**Word stress.** Trained and exported for reproducibility, not loaded by the
runtime. The corpus scores stress `{5, 10}` and only 0.9% of words score 5, and
mono-syllable words are correct by definition, so the positive class is both
tiny and only reachable in polysyllables. The head reaches 0.044 precision on
test — about twenty false alarms per real one. 0.5-community's per-character
features raise it to 0.053, which is the right direction and nowhere near
enough. The limit is the label: 127 positives in the test split, none of them
reachable in a mono-syllable.

**Prosodic.** Trained, exported, not wired to an output field. There is nowhere
in the current API contract for it.

## Known limitations

**Prompt length.** Corpus utterances are 1–10 words, median 6. Read Aloud
prompts run 40–80. The utterance heads were trained without the features whose
magnitude tracks prompt length — `duration`, `spokenSec`, `pauseTotal`,
`nPause`, `nRef`, `nAligned`, `nGopWords` — because those are the strongest
predictors on this corpus and would be dominated by prompt length at the target
length. This costs little on the corpus and should transfer better, but it is a
design argument, not a measurement: nothing here has been evaluated at 40–80
words.

**Speaker variance.** A 250-utterance slice of the test split scores well away
from the full split — speaker-to-speaker variance on this corpus is large, and
small samples are not informative about a model's quality. Every number here is
on all 2500 test utterances.

**Speaker population.** All speakers are Mandarin-native, and roughly half are
children. Behaviour on other first languages is untested.

**Recording conditions.** Signal-level features (`rmsMean`, `rmsStd`,
`rmsPeak`, `activeRatio`) are excluded. Each speaker was recorded once, so
recording conditions track the speaker and the speaker tracks proficiency; a
head that leaned on gain would look good for the wrong reason. Removing all four
changed test correlation by at most 0.008 in either direction.

**Reading, not speaking.** Everything assumes the speaker is reading a known
reference text. There is no free-speech mode.

## How 0.6 was promoted

0.6 reads a tensor the previous acoustic asset did not emit, so the release and
the checkpoint were one change across two artifacts, done in this order:

1. `pnpm models:export` — rebuild the graph with `logits` + `hidden`. It refuses
   to write a file that disagrees with the PyTorch pipeline, the export is
   byte-reproducible, and its `logits` come out bit-identical to the previous
   graph's, so nothing that already worked could move.
2. Upload it under a **new** release tag, `acoustic-v2`. The pin lives in
   `scripts/download-models.ts`, so replacing the file on the old tag would have
   made every existing checkout delete its download and fail; each graph keeping
   its own tag is what leaves older clones alone.
3. Update the pin and the URL in `scripts/download-models.ts`, and `sha256` /
   `hidden_layer` in `models/labels.json`.
4. Point `releases/CURRENT` at `0.6-community`.

Order matters in one direction only. 4 before 1–3 leaves every default install
unable to score. 1–3 without 4 is harmless: a graph that emits a hidden layer
serves 0.4 and 0.5 unchanged, because they never ask for it.

## Reproducing

```bash
pnpm models:export                   # acoustic ONNX, logits + hidden layer 3
pnpm corpus:download
pnpm corpus:prepare
pnpm features:extract
python training/sweep_weak_char.py   # derives WEAK_CHAR_GOP from this corpus
pnpm features:extract --hidden training/data/hidden_l3.f32
pnpm hidden:fit                      # fits the projection on the train split
pnpm heads:train --features training/data/features_v3.jsonl --feature-version 3
pnpm heads:export
pnpm heads:eval --features training/data/features_v3.jsonl   # the numbers above
pnpm runtime:verify    # word bands through the Node serving path
pnpm scores:verify     # utterance scores through the same path
```

The extraction is run twice on purpose: `WEAK_CHAR_GOP` is swept on the first
pass and is an input to the second. The hidden sidecar is written by the same
extractor that serves requests, so a head is fitted on exactly the columns it
will later be served — checked directly, and the two agree bit for bit.

The sweep runs before the final extraction because `WEAK_CHAR_GOP` feeds two
word features and two utterance aggregates. It is swept on the training split
only, and lands at **-4.3** on this corpus; the objective is flat between about
-5 and -2.5, so the value matters less than deriving it here rather than
importing one measured on other data.

`heads:eval` scores the ONNX graphs in Python. `runtime:verify` runs corpus
audio through the whole Node path and checks it lands on the same answers, which
is what stops training and serving from drifting apart.

## Where these live

Checked into `releases/`, weights included, and loaded from there without a
download — `releases/CURRENT` names the generation in use. Every generation is
kept rather than only the newest, so the comparison above can be re-run:

```bash
training/.venv/bin/python training/eval.py --onnx releases/0.4-community --split test
training/.venv/bin/python training/eval.py --onnx releases/0.5-community --split test
```

## License

The weights are CC BY 4.0, matching the corpus they derive from — see
`LICENSE-MODELS` for the attribution to reproduce, and `NOTICE` for the corpus
citation. Code in this repository is Apache-2.0 (`LICENSE`). The acoustic model
is a third-party checkpoint under its own license; see `docs/models.md`.
