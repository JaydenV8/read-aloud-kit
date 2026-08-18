# Development

```bash
pnpm install
pnpm models:download
pnpm test
pnpm oxlint
pnpm oxfmt
pnpm dev
```

## Edit constants

`MIN_OMIT_RUN`, `MIN_EDGE_OMIT`, `MIN_INSERT_RUN` and the near-phone rules in
`@readaloudkit/edits`, plus `GOP_SILENT` in `@readaloudkit/gop`, set how
forgiving the analysis is. They are calibrated on a small number of utterances.
Retune them against a corpus with independent per-word labels, not against the
regression fixtures — those exist to detect change, and tuning until they pass
makes them useless.

## Fixtures

`tests/fixtures/` holds goldens generated from a reference Python implementation
on synthetic audio: a planted CTC lattice, a hand-built alignment, and a
deterministic tone. `examples/sample.wav` is synthetic for the same reason and
is reproducible byte for byte — see `examples/README.md`. The tone is defined by a formula that both languages
evaluate rather than by a stored sample array, so neither side can drift by
copying the other's numbers.

Adding a member to `WORD_FEATURE_KEYS_V2`, `UTTERANCE_GOP_KEYS_V2` or
`UTTERANCE_FEATURE_KEYS` means adding it to the fixture in the same change. A
feature with no cross-checked golden can silently diverge from what a scoring
head was trained on.

Never renumber or reorder an existing key. A shipped release names the keys it
was fitted on and the runtime looks them up by name, so appending is safe and
renaming quietly breaks every release that came before.
