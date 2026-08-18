# Examples

## `sample.wav`

Synthetic speech, not a recording of a person. 16 kHz mono 16-bit, 2.57 s,
reading the line used throughout the documentation:

> The university provides many opportunities.

It is macOS speech synthesis, and the file in this directory is reproducible
byte for byte on a Mac with the default system voice:

```bash
say -o sample.aiff "The university provides many opportunities."
afconvert -f WAVE -d LEI16@16000 -c 1 sample.aiff sample.wav
```

Synthetic audio is used deliberately. Read-aloud scoring works on recordings of
people reading a script, and a sample checked into a public repository would be
one person's voice published for as long as the repository exists. The
regression goldens in `tests/fixtures/` are synthetic for the same reason.

A consequence worth knowing when reading the output: the analyzer scores this
file well because a speech synthesiser articulates evenly. It exercises the
pipeline; it is not a demonstration of how the model behaves on a learner.
