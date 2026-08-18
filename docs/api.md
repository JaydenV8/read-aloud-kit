# API

Base URL: `http://127.0.0.1:3000`

## `GET /health`

```json
{
  "ok": true,
  "ready": true,
  "package": "0.1.0",
  "acoustic": "community",
  "scoring": { "backend": "community", "version": "0.5-community" }
}
```

`scoring.version` names the weights that are loaded. The backend name alone does
not — every generation answers to `community` — so quote the version in a bug
report. With no heads installed it reads `{ "backend": "none", "version": null }`.

## `POST /v1/read-aloud/analyze`

`multipart/form-data`

| field | required | notes |
|---|---|---|
| `audio` | yes | WAV preferred; other types need system ffmpeg |
| `referenceText` | yes | English read-aloud prompt |

`text` is accepted as a legacy alias for `referenceText`.

Limits: 20 MB, 0.2–120 s, reference text up to 5000 characters. Audio is not written to a durable store.

### Response

```jsonc
{
  "version": { "api": "v1", "package": "0.1.0" },
  "reference": "The university provides many opportunities for research and study.",
  "hypothesis": "the university provides opportunities for search and study and study",
  "durationMs": 3762,

  "words": [ /* one per reference word, plus insertions and repetitions */ ],
  "edits": [ /* the non-correct subset of words, with reference/hypothesis indices */ ],
  "analysis": { "omissions": 1, "substitutions": 1, "insertions": 1, "repetitions": 0 },

  "content":  { "score": 3.33, "maxScore": 5, "mode": "calibrated",
                "strict": 2.78, "calibrated": 3.33,
                "referenceWords": 9, "chargedErrors": 3 },
  "prosody":  { "wpm": 199.1, "coverage": 0.889, /* 21 features in total */ },
  "pauses":   [ { "startMs": 2595, "endMs": 3199, "durMs": 604,
                  "where": "mid", "afterWordIndex": 7 } ],
  "tips":     [ "Barely spoken: many. Every omitted word costs a content point." ],

  "scores": { "backend": "community", "content": 3.33,
              "pronunciation": 71.1, "fluency": 78.7, "overall": 70.8 }
}
```

With no scoring backend installed, `backend` is `none` and the three
model-derived scores are `null`; every other field is returned either way.

#### `words[]`

| field | notes |
|---|---|
| `reference` / `hypothesis` | `null` for an insertion / an omission respectively |
| `status` | `correct` \| `omission` \| `substitution` \| `insertion` \| `repetition` |
| `statusEvidence` | `edit` if the transcript comparison decided it, `acoustic` if the aligned span had no speech in it |
| `startMs` / `endMs` | from forced alignment; `null` for words with no reference position |
| `confidence` | `sigmoid(marginMean)` in 0–1. 0.5 means the aligned character only ties with its strongest competitor. Not a pronunciation score |
| `gop` | per-word acoustic evidence: `tok`, `t0`, `t1`, the per-character series in `chars`, and the 36 numeric features named by `WORD_FEATURE_KEYS_V2` that the heads consume |
| `level` | `good` / `average` / `bad` from a scoring backend, `null` without one |
| `needsAttention` | whether the same head thinks this word is worth pointing at |

`level` and `needsAttention` come off one threshold, so a word is never shown as
`bad` while the flag says it is fine. **Read `needsAttention`, not `level`**: the
flag runs precision 0.47 on held-out data, while telling `average` from `bad`
runs about 0.28 for `average`. Both are hints. `MODEL_CARD.md` has the numbers.

#### `content`

`strict` charges every omission, substitution and insertion, which is how the
published Read Aloud rule reads. `calibrated` forgives near-phone and
function-word substitutions plus very short omission runs, then charges back the
omissions the acoustics contradict. `score` is whichever mode is active
(`calibrated` by default). Neither is an official PTE content score.

#### `prosody`

21 utterance features: `duration`, `nRef`, `nAligned`, `coverage`, `spokenSec`,
`wpm`, `nPause`, `pauseTotal`, `pauseMax`, `pauseMean`, `pauseRatio`,
`wordDurMean`, `wordDurStd`, `wordDurCv`, `leadSil`, `tailSil`, `alignOk`,
`rmsMean`, `rmsStd`, `rmsPeak`, `activeRatio`.

`wpm` is an articulation rate: words over speaking time, pauses excluded.
`coverage` counts words actually spoken, not words the aligner placed.

#### `scores`

`content` is always present because it comes from a rule, and is on 0–5.
`pronunciation`, `fluency` and `overall` come from the scoring heads on a 10–90
range; `backend` names the one in use (`none` or `community`). The heads ship in
`releases/`, so these are populated from a clone — they go `null` only if you
remove them.

The 10–90 range is a linear rescaling of a 0–10 corpus score, chosen because it
is the range readers expect. It is **not** an official examination score and is
calibrated against nothing. See `MODEL_CARD.md`.

### Errors

Request parameters are validated with zod before any audio is decoded. A rejected request returns `400` with every failing field:

```json
{
  "ok": false,
  "error": "referenceText is required",
  "issues": [{ "field": "referenceText", "message": "referenceText is required" }]
}
```

`503` means the community acoustic model is not installed — run `pnpm models:download`. That check runs before the body is read, so an unconfigured server rejects uploads immediately.

Anything that goes wrong inside returns a fixed body and logs the detail server-side:

```json
{ "ok": false, "error": "analyze failed", "code": "internal" }
```

Only errors raised deliberately with a status — bad audio, a missing field, a
file over the limit — carry their own message back to the caller. Library
failures do not: onnxruntime, for one, quotes the absolute path of the model
file when a load fails, and that is not the caller's business.
