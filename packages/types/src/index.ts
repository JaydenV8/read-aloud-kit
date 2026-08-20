export const PACKAGE_VERSION = '0.1.0'

export type AcousticOutput = {
  logits: Float32Array
  frames: number
  vocabSize: number
  /**
   * An intermediate transformer layer, `frames * hiddenSize`, when the graph
   * emits one.
   *
   * The logits are what the decode and the alignment read. They are also the
   * end of a pipeline trained to keep only what separates one character from
   * another, and measured against expert labels an early layer carries more of
   * what the scoring heads want than the last one does. A graph that emits both
   * gives the heads that layer for the cost of the copy, since the forward pass
   * computes it either way. Absent on a single-output model.
   */
  hidden?: Float32Array
  hiddenSize?: number
  hiddenLayer?: number
}

export interface AcousticModel {
  infer(audio: Float32Array): Promise<AcousticOutput>
}

export type AlignmentSpan = {
  tokenId: number
  startFrame: number
  endFrame: number
  score: number
}

/**
 * Per-word acoustic evidence from the forced-alignment path. The field order of
 * `WORD_FEATURE_KEYS` in `@readaloudkit/gop` is the model input contract; adding
 * a member here without updating that list will not reach a scoring head.
 */
/**
 * One aligned character, before it is summarised into a word.
 *
 * Kept because averaging it away loses what a listener reacts to: two words can
 * share a mean posterior while one was read cleanly and the other had a single
 * mangled syllable.
 */
export type GopChar = {
  lab: string
  gop: number
  margin: number
  dur: number
}

export type GopWord = {
  tok: string
  t0: number
  t1: number
  /**
   * The same span in frames, half-open. `t0`/`t1` are rounded to milliseconds
   * for display, and anything that has to index back into a per-frame tensor
   * needs the indices themselves rather than a reconstruction of them.
   */
  f0: number
  f1: number
  gopMean: number
  gopMin: number
  gopStd: number
  marginMean: number
  marginMin: number
  nFrames: number
  nChars: number
  dur: number
  charPerSec: number
  blankRatio: number
  logNFrames: number

  /** The per-character series the summary above was built from. */
  chars: GopChar[]

  // Shape of the per-character posterior series.
  charGopMin: number
  charGopMax: number
  charGopStd: number
  charGopRange: number
  charGopP25: number
  /** Where the worst character sits: 0 at the start of the word, 1 at the end. */
  worstCharPos: number
  nWeakChars: number
  fracWeakChars: number
  headGop: number
  tailGop: number
  headTailDelta: number

  // Shape of the per-character duration series, which is the stress proxy.
  // English stress lives in how duration is distributed across a word, so a
  // word-level duration average cannot represent it at all.
  charDurMax: number
  charDurMin: number
  charDurStd: number
  charDurCv: number
  longestCharPos: number
  durFrontBackRatio: number

  // Margin distribution inside the word.
  charMarginMin: number
  charMarginP25: number

  // Context: a word after a long silence is heard differently from one in flow.
  gapBefore: number
  gapAfter: number
  prevGopMean: number
  nextGopMean: number
  relDur: number
  relCharPerSec: number
}

export type PauseWhere = 'lead' | 'mid' | 'tail'

export type PauseSpan = {
  startMs: number
  endMs: number
  durMs: number
  where: PauseWhere
  /** Index into `words` that this pause follows, so a reader can place a break mark. */
  afterWordIndex: number | null
}

/**
 * Utterance-level prosody. `UTTERANCE_FEATURE_KEYS` in `@readaloudkit/features`
 * pins the order used for model input.
 */
export type ProsodyFeatures = {
  duration: number
  nRef: number
  nAligned: number
  coverage: number
  spokenSec: number
  wpm: number
  nPause: number
  pauseTotal: number
  pauseMax: number
  pauseMean: number
  pauseRatio: number
  wordDurMean: number
  wordDurStd: number
  wordDurCv: number
  leadSil: number
  tailSil: number
  alignOk: number
  rmsMean: number
  rmsStd: number
  rmsPeak: number
  activeRatio: number
}

export type WordStatus = 'correct' | 'omission' | 'substitution' | 'insertion' | 'repetition'

/**
 * `edit` means the reference/hypothesis alignment produced this status.
 * `acoustic` means the edit alignment called the word spoken but the aligned
 * span carries almost no speech, so it was re-marked as an omission.
 */
export type StatusEvidence = 'edit' | 'acoustic'

export type WordLevel = 'good' | 'average' | 'bad'

export type AnalysisWord = {
  reference: string | null
  hypothesis: string | null
  status: WordStatus
  statusEvidence: StatusEvidence
  confidence: number | null
  startMs: number | null
  endMs: number | null
  gop: GopWord | null
  /**
   * Pronunciation band from a scoring backend, `null` without one. Reported for
   * parity with three-colour displays, but the three-way call is imprecise:
   * check the model card before rendering `average` and `bad` differently.
   */
  level: WordLevel | null
  /**
   * The same head collapsed to one decision at a calibrated threshold. This is
   * the signal a display should highlight — it is roughly twice as precise as
   * telling `average` from `bad`. Still a hint, not a verdict.
   */
  needsAttention: boolean | null
}

export type AnalysisEdit = {
  type: WordStatus
  referenceWord: string | null
  hypothesisWord: string | null
  referenceIndex: number | null
  hypothesisIndex: number | null
  confidence: number | null
}

/**
 * `strict` charges every omission, substitution and insertion the way the
 * published Read Aloud rule does. `calibrated` additionally forgives near-phone
 * and function-word substitutions plus very short omission runs, then adds back
 * omissions that the acoustics contradict.
 */
export type ContentMode = 'calibrated' | 'strict'

export type ContentScore = {
  score: number
  maxScore: number
  mode: ContentMode
  strict: number
  calibrated: number
  referenceWords: number
  chargedErrors: number
}

export type Scores = {
  backend: string
  content: number | null
  pronunciation: number | null
  fluency: number | null
  overall: number | null
}

export type ScoredWord = {
  level: WordLevel
  needsAttention: boolean
}

export type ScoringResult = {
  backend: string
  pronunciation?: number
  fluency?: number
  overall?: number
  content?: number
  /** One entry per `ScoringInput.gopWords`, in the same order. */
  words?: ScoredWord[]
}

/**
 * An intermediate acoustic layer, pooled.
 *
 * Kept off `AnalysisWord` on purpose: this is hundreds of floats per word, it
 * means nothing to a caller, and the HTTP response is not the place for it.
 * It travels from the analyzer to the scoring backend and stops there.
 */
export type PooledHidden = {
  /** One vector of `size` per aligned word, in `gopWords` order. */
  words: Float32Array[]
  /** The same pooling over the whole clip. */
  utterance: Float32Array
  size: number
  /** Which transformer layer, 1-based, or null if the graph did not say. */
  layer: number | null
}

export type ScoringInput = {
  durationMs: number
  words: AnalysisWord[]
  gopWords: GopWord[]
  hypothesis: string
  reference: string
  prosody: ProsodyFeatures
  content: ContentScore
  /** Present only when the acoustic graph emits a hidden layer. */
  hidden?: PooledHidden
}

export interface ScoringBackend {
  readonly name: string
  /**
   * Which weights are loaded, e.g. `0.5-community`. The backend name alone does
   * not identify them — successive releases share it — so a bug report has no
   * way to say which numbers it saw unless this is reported too.
   */
  readonly version?: string
  score(input: ScoringInput): Promise<ScoringResult | null>
}

export type ReadAloudAnalysis = {
  version: { api: 'v1'; package: string }
  reference: string
  hypothesis: string
  durationMs: number
  words: AnalysisWord[]
  edits: AnalysisEdit[]
  analysis: {
    omissions: number
    substitutions: number
    insertions: number
    repetitions: number
  }
  content: ContentScore
  prosody: ProsodyFeatures
  pauses: PauseSpan[]
  tips: string[]
  scores: Scores
  /** Only when `includeHidden` was asked for; never set by the HTTP route. */
  hidden?: PooledHidden
}
