# Limitations

### ASR / substitution

CTC may label accent, near-phones, or decode errors as substitutions. A decoded substitution does not necessarily mean the speaker said a different word.

### Content

`content.strict` applies the published one-error-per-word rule to a machine
transcript, so it inherits every ASR error above. `content.calibrated` softens
those cases but then has to decide which softened omissions to charge back from
the acoustics. Neither is an official PTE content score, and neither is graded
against an official maximum that varies with prompt length.

### Acoustic omissions

`statusEvidence: "acoustic"` comes from a threshold on the mean posterior of the
aligned span. It is calibrated on a handful of utterances, not on a corpus with
independent per-word pronunciation labels. A word pronounced badly enough could
in principle cross it and be reported as unread. Pass `acousticOmissions: false`
to report only what the transcript comparison found.

### Word bands

`needsAttention` runs precision 0.47 and recall 0.54 on held-out data: nearly one
flag in two is real, and about half the real problems get flagged. Treat it as
"worth listening to again", never as "you said this wrong". Separating `average`
from `bad` is weaker, near 0.28 precision for `average`.

### Pronunciation / fluency

With the community heads installed these come from models trained on
speechocean762 and rescaled to 10–90. They correlate 0.65–0.73 with expert
scores on that corpus's test split. They are **not** Pearson's scores, the 10–90
mapping is calibrated against nothing, and no evaluation has been run at Read
Aloud prompt length. `MODEL_CARD.md` has the full picture.

### Prosody

`wpm` is an articulation rate — words over speaking time, pauses excluded — so
it reads higher than a wall-clock words-per-minute. Pauses, microphone, noise,
rate and ASR errors all move the prosody features.

### Tips

Tips are rule-driven and only describe what the analysis measured. Lines about
per-word pronunciation and weak forms need a scoring backend and stay silent
without one.

### Pearson

ReadAloudKit does not reproduce Pearson's proprietary scoring system.

### Prompt length

The edit and pause rules were tuned against Read Aloud prompts of roughly 40–80
words. On a very short prompt the adaptive pause threshold has too few gaps to
estimate a spread from, and a single edit moves the content score by a large
step, so both read as noisier than they do at full length.
