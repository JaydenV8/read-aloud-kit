# Architecture

HTTP: Hono on Node.js. Inference: TypeScript + `onnxruntime-node`. There is no Python in the public request path.

```text
Client
  → Hono
  → WAV / ffmpeg
  → ONNX wav2vec2 CTC  (one session, one forward)
  → log-softmax
      ├─ greedy collapse → hypothesis
      └─ CTC Viterbi → path → GOP spans
  → word edits (soften) + acoustic omission check
  → prosody / pauses / content / tips
  → optional scoring backend
  → JSON
```

`ReadAloudAnalyzer` lives in `@readaloudkit/core`. Routes only validate and serialize.

## Packages

| package | role |
|---|---|
| `types` | shared shapes; nothing else depends on anything but this |
| `audio` | WAV decode, mono, 16 kHz, optional ffmpeg shell-out |
| `inference` | ONNX session for the acoustic model |
| `ctc` | greedy decode, reference tokenisation, log-softmax |
| `alignment` | CTC Viterbi forced alignment |
| `gop` | per-word acoustic evidence and the word feature vector |
| `edits` | reference-vs-hypothesis alignment, softening, content arithmetic |
| `features` | utterance prosody and the utterance feature vector |
| `prosody` | adaptive pause detection |
| `tips` | rule-driven advice from the error list |
| `scoring` | pluggable scoring backend; none ships yet |
| `core` | the analyzer that wires the above together |

## Two evidence channels

The reference-vs-hypothesis edit alignment and the acoustics disagree in a
useful way, so both are kept.

The edit alignment forgives a single skipped short word, because a transcript
comparison alone cannot tell a skipped word from a decode slip. The forced
alignment can: it has to place every reference word somewhere, so a skipped word
gets squeezed into a sliver between its neighbours and its mean posterior falls
an order of magnitude. `words[].statusEvidence` records which channel decided
the status.

The same asymmetry is why `coverage` is computed from words actually spoken
rather than words the aligner placed — the aligner always places all of them.

## Feature contracts

`WORD_FEATURE_KEYS_V2` and `UTTERANCE_GOP_KEYS_V2` (`@readaloudkit/gop`), with
`UTTERANCE_FEATURE_KEYS` (`@readaloudkit/features`), pin the order a scoring head
is trained on. The v1 lists they superseded are still exported, because a release
fitted on those still loads. Training extracts features through these same
packages, so there is no separate implementation to drift against at serving
time.

A release records the key list it was fitted on and the runtime assembles its
input from that rather than from a version number, which is why one code path
serves both `0.4-community` and `0.5-community`. See `docs/models.md`.
