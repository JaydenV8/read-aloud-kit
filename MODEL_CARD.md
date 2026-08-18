# Model card — ReadAloudKit community heads

Pronunciation and fluency heads for read-aloud practice. They turn the acoustic
features in `@readaloudkit/gop` and `@readaloudkit/features` into a per-word
band and three utterance scores.

**These are not PTE scores.** They are not produced by, endorsed by, or
comparable to Pearson's scoring system. See [Scale](#scale).

| | |
|---|---|
| Version | 0.5-community |
| Model | LightGBM, exported to ONNX |
| Size | 1.0 MB for the four shipped heads |
| Training data | speechocean762 only |
| Word features | 36 (`WORD_FEATURE_KEYS_V2`) |
| Utterance features | 32 of 45 |
| Runtime | onnxruntime-node, CPU |

Releases are numbered to match the generation they correspond to, with a
`-community` suffix. `0.4-community` summarised each word as eleven averages;
`0.5-community` keeps the shape of its per-character series. Both are in
`releases/`, and a release carries the feature contract it was fitted on, so the
runtime assembles its input from the shipped key list rather than from a version
number.

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
| good | 0.936 | 0.918 | 0.927 | 14062 |
| average | 0.281 | 0.241 | 0.259 | 1010 |
| bad | 0.351 | 0.514 | 0.417 | 895 |

Accuracy 0.853, macro F1 0.535.

The single decision that matters — is this word worth pointing at — runs
**precision 0.472, recall 0.539**, flagging 13.6% of words against a true rate of
11.9%.

So: nearly one flag in two is real, and about half of the real problems get
flagged. That is a usable hint and a bad verdict. A display should treat a flag
as "worth listening to again", never as "you said this wrong". Telling `average`
from `bad` is still weak — `average` sits near 0.28 precision — so consider
rendering one highlight rather than two shades.

### Against 0.4-community

Same corpus, same official split, same decision rule, same harness. The only
change is what a word is summarised as.

| | 0.4-community | 0.5-community |
|---|---|---|
| word accuracy | 0.831 | **0.853** |
| word macro F1 | 0.511 | **0.535** |
| `bad` F1 | 0.357 | **0.417** |
| flag precision | 0.417 | **0.472** |
| flag recall | 0.562 | 0.539 |
| words flagged | 2564 | **2174** |
| accuracy r | 0.652 | **0.665** |
| fluency r | 0.729 | **0.737** |
| total r | 0.684 | **0.687** |

The gain is concentrated where it is worth having: the flag gets more precise
while flagging 390 fewer words, and `bad` — the rarest and most consequential
class — improves most. The utterance heads move much less, which is what a
six-word corpus predicts: those features describe how delivery is distributed
across an utterance, and six words leave little to distribute.

### Utterance scores

2500 utterances, Pearson correlation against the expert score.

| head | r | MAE | MAE if you just predicted the mean |
|---|---|---|---|
| accuracy → `scores.pronunciation` | 0.652 | 0.865 | 1.131 |
| fluency → `scores.fluency` | 0.729 | 0.706 | 1.056 |
| total → `scores.overall` | 0.684 | 0.827 | 1.182 |

For context, GOPT (ICASSP 2022) reports 0.742 sentence-level PCC on this corpus
with a transformer over per-phone GOP vectors. These heads are gradient-boosted
trees over 19 utterance-level features and land below that, which is the
expected trade for a 1.0 MB model that runs on CPU in Node.

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

## Reproducing

```bash
pnpm corpus:download
pnpm corpus:prepare
pnpm features:extract
python training/sweep_weak_char.py   # derives WEAK_CHAR_GOP from this corpus
pnpm features:extract                # again, once the constant is set
pnpm heads:train
pnpm heads:export
pnpm heads:eval        # the numbers above
pnpm runtime:verify    # the same heads through the Node serving path
```

The sweep runs before the final extraction because `WEAK_CHAR_GOP` feeds two
word features and two utterance aggregates. It is swept on the training split
only, and lands at **-4.3** on this corpus; the objective is flat between about
-5 and -2.5, so the value matters less than deriving it here rather than
importing one measured on other data.

`heads:eval` scores the ONNX graphs in Python. `runtime:verify` runs corpus
audio through the whole Node path and checks it lands on the same answers, which
is what stops training and serving from drifting apart.

## License

Code Apache-2.0. The training corpus is CC BY 4.0 and requires the attribution
above; see `NOTICE`. The acoustic model is a third-party checkpoint under its own
license; see `docs/models.md`.
