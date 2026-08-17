# Training

Python lives here and nowhere else. The serving path is TypeScript; this
directory only produces the weights it loads.

```bash
pnpm corpus:download     # fetch + verify + unpack speechocean762
pnpm corpus:prepare      # corpus layout -> data/utterances.jsonl
pnpm features:extract    # utterances -> data/features.jsonl  (~15 min)
```

`data/` is gitignored. Nothing in it is redistributable from here.

## Corpus

[speechocean762](https://www.openslr.org/101/), **CC BY 4.0**.

> speechocean762: An Open-Source Non-Native English Speech Corpus for
> Pronunciation Assessment. Interspeech 2021.

CC BY obliges attribution when a model or its output is shared publicly, not
when it is merely used for training. Publishing weights trained here therefore
requires the citation above in `NOTICE` and in the model card. CC BY carries no
ShareAlike clause, so the weights themselves may be licensed separately.

## What the corpus actually contains

Measured from the archive, not from its README — the two disagree in three
places, noted below.

| | |
|---|---|
| Utterances | 5000, split 2500 train / 2500 test |
| Speakers | 125 per split, **no overlap**; the official split is usable as-is |
| Age | 6–43. Adults (18+) read 1340 of train, 1220 of test |
| Audio | 16 kHz mono 16-bit, median 3.5 s |
| Words per utterance | 1–10, median 6 |
| Annotators | five experts, independently |

Where the corpus README and the archive disagree, the archive wins:

- `scores.json` is under `resource/`, not at the top level
- sentence `completeness` in `scores.json` is scaled 0–10; the 0.0–1.0 range in
  the README describes `scores-detail.json`
- a word's `phones` is a list, not a space-separated string

### Label distributions

| Level | Field | Range | Shape |
|---|---|---|---|
| Sentence | accuracy | 0–10 | mean 7.66, median 8 |
| Sentence | fluency | 0–10 | mean 7.78, median 8 |
| Sentence | prosodic | 0–10 | mean 7.46, median 8 |
| Sentence | total | 0–10 | mean 7.22, median 8 |
| Sentence | completeness | 0–10 | **mean 9.98** — effectively constant |
| Word | accuracy | 0–10 | 88.8% are 10 |
| Word | stress | {5, 10} | **0.9% are 5** |
| Word | total | 0–10 | 87.2% are 10 |
| Phone | accuracy | 0–2 | |

Three of these shape the plan more than the averages do.

**`completeness` is not usable as a content label.** At mean 9.98 it carries
almost no signal, which is what you would expect: these speakers were reading a
short prompt and generally read all of it. Content is computed from a rule in
`@readaloudkit/edits` instead, and needs no label.

**`stress` is a 1-in-110 positive class.** A classifier trained on it will need
heavy reweighting and will still be the weakest head here. Treat any weak-form
output as a hint, and say so wherever it is displayed.

**Utterances are 1–10 words.** Read Aloud prompts run 40–80. Utterance-level
prosody is the part that suffers: a six-word utterance has at most five gaps, so
`nPause`, `pauseMax` and `wordDurCv` are estimated from almost nothing, and
`wpm` — an articulation rate — reads far higher than it would on a full prompt.
Word-level and phone-level labels do not have this problem, which is why the
word heads come first and the utterance heads are treated as the uncertain ones.

## Schema

`prepare.py` writes a corpus-neutral `utterances.jsonl`; nothing downstream
knows where the data came from. A second corpus needs another script shaped like
`prepare.py` and nothing else.

Word `level` is binned in `prepare.py`, not in the trainer, so the binning is
visible in the data. `data/prepare_report.json` carries the histogram it was
chosen from.

## Feature extraction

`extract_features.ts` calls the same `ReadAloudAnalyzer.analyze` the HTTP API
calls. Training features and serving features are the same code path by
construction rather than by discipline — a second extraction path is the classic
way to ship a model that scores differently in production than in evaluation,
and it fails silently.

`data/features.meta.json` records the feature order the weights are bound to.
Regenerate features whenever `WORD_FEATURE_KEYS` or `UTTERANCE_FEATURE_KEYS`
changes.

## What the features support

Univariate correlation against the expert labels, over all 5000 utterances and
31816 words. Single features, no model.

| Target | Strongest features |
|---|---|
| word accuracy | `gopMin` +0.39, `gopStd` −0.37, `marginMin` +0.37 |
| sentence accuracy | `pauseTotal` −0.57, `duration` −0.56, `marginMean` +0.53 |
| sentence fluency | `pauseTotal` −0.70, `duration` −0.68, `pauseRatio` −0.65 |
| sentence prosodic | `pauseTotal` −0.67, `duration` −0.65, `pauseRatio` −0.62 |

The word-level picture is what it should be: goodness-of-pronunciation carries
the pronunciation label, and the worst frame in a word (`gopMin`) says more than
its average.

The sentence-level picture contains a trap. `pauseTotal` and `duration` are the
strongest fluency predictors here, and both scale with how long the utterance
is. On a six-word prompt that is a real fluency signal — a hesitant reader takes
longer. On a 40–80 word prompt, duration is dominated by prompt length instead,
and a head that leaned on it would transfer badly to the length this project
actually targets.

`pauseRatio` is nearly as strong at −0.65 and is length-invariant, and `wpm` is
a rate rather than a total. Prefer the normalized forms when training the
utterance heads, and check what the model leans on before believing the score.

## Provenance rules

Weights published from this directory must be reproducible from this directory.
That means:

1. Training labels come only from corpora named here.
2. Validation and early stopping use the same corpora. Selecting a model against
   held-out data from somewhere else leaks that source into the weights just as
   surely as training on it would.
3. No constant fitted elsewhere is copied in — no coefficients, no percentile
   tables, no thresholds.

Each artifact ships a `provenance.json` recording the corpora, their checksums,
and the seed, so the claim can be checked rather than trusted.
