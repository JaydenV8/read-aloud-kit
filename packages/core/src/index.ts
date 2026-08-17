import { decodeWav, maybeFfmpegToWav, toMono16k } from '@readaloudkit/audio'
import { alignReference } from '@readaloudkit/alignment'
import {
  greedyIds,
  idsToWords,
  labelIndex,
  logSoftmaxRows,
  tokenizeReference,
} from '@readaloudkit/ctc'
import { contentFromEdits, dualTrackTokens, editsFromTexts } from '@readaloudkit/edits'
import { acousticFeatures, pausesFromWords, prosodyFromAlignment } from '@readaloudkit/features'
import { wordConfidence, wordIsOmission, wordsFromPath } from '@readaloudkit/gop'
import { acousticReady, getAcousticModel } from '@readaloudkit/inference'
import { displayPauses } from '@readaloudkit/prosody'
import { loadScoringBackend } from '@readaloudkit/scoring'
import { buildTips } from '@readaloudkit/tips'
import {
  PACKAGE_VERSION,
  type AnalysisEdit,
  type AnalysisWord,
  type ContentMode,
  type ContentScore,
  type GopWord,
  type PauseSpan,
  type ReadAloudAnalysis,
  type ScoringBackend,
  type Scores,
  type WordStatus,
} from '@readaloudkit/types'

export type AnalyzeInput = {
  audio: Uint8Array
  referenceText: string
  mime?: string
  /** Defaults to `calibrated`; see `ContentMode`. */
  contentMode?: ContentMode
  /** Set false to report only what the edit alignment found. */
  acousticOmissions?: boolean
}

const KIND_STATUS: Record<string, WordStatus> = {
  word: 'correct',
  omission: 'omission',
  replacement: 'substitution',
  insert: 'insertion',
  repeat: 'repetition',
}

const MAX_AUDIO_BYTES = 20 * 1024 * 1024
const MIN_DURATION_MS = 200
const MAX_DURATION_MS = 120_000

export class ReadAloudAnalyzer {
  private readonly scoring: Promise<ScoringBackend>

  constructor(scoring?: ScoringBackend) {
    this.scoring = scoring ? Promise.resolve(scoring) : loadScoringBackend()
  }

  ready(): boolean {
    return acousticReady()
  }

  /**
   * The installed scoring backend: its name, and which weights it loaded.
   * `none` with a null version when nothing is wired up.
   */
  async scoringBackend(): Promise<{ backend: string; version: string | null }> {
    const s = await this.scoring
    return { backend: s.name, version: s.version ?? null }
  }

  async analyze(input: AnalyzeInput): Promise<ReadAloudAnalysis> {
    const ref = (input.referenceText || '').trim()
    if (!ref) throw Object.assign(new Error('referenceText is required'), { status: 400 })
    if (!input.audio?.byteLength)
      throw Object.assign(new Error('audio is required'), { status: 400 })
    if (input.audio.byteLength > MAX_AUDIO_BYTES) {
      throw Object.assign(new Error('audio larger than 20MB'), { status: 400 })
    }

    const wavBytes = await maybeFfmpegToWav(input.audio, input.mime ?? 'audio/wav')
    const pcm = toMono16k(decodeWav(wavBytes))
    if (pcm.durationMs < MIN_DURATION_MS)
      throw Object.assign(new Error('audio shorter than 0.2s'), { status: 400 })
    if (pcm.durationMs > MAX_DURATION_MS)
      throw Object.assign(new Error('audio longer than 120s'), { status: 400 })
    const duration = pcm.durationMs / 1000

    const model = await getAcousticModel()
    const acoustic = await model.infer(pcm.samples)
    const labels = model.meta.labels
    const blank = model.meta.blank ?? 0
    const logProbs = logSoftmaxRows(acoustic.logits, acoustic.frames, acoustic.vocabSize)
    const greedy = greedyIds(acoustic.logits, acoustic.frames, acoustic.vocabSize)
    const hypWords = idsToWords(greedy, labels, blank)
    const hypothesis = hypWords.join(' ')

    const dict = labelIndex(labels)
    const targetIds = tokenizeReference(ref, dict)
    const { path } = alignReference({
      logProbs,
      frames: acoustic.frames,
      vocabSize: acoustic.vocabSize,
      referenceTokens: targetIds,
      blankId: blank,
    })
    const gopWords = wordsFromPath(
      path,
      logProbs,
      acoustic.frames,
      acoustic.vocabSize,
      labels,
      blank,
      duration,
    )
    const { tokens, counts } = dualTrackTokens<GopWord>(gopWords, ref, hypWords)

    // The edit alignment forgives short omission runs because transcript-only
    // evidence is unreliable for one skipped word. The acoustics are not: a word
    // the aligner parked on silence was not read, whatever the decode said.
    const useAcoustic = input.acousticOmissions !== false
    const words: AnalysisWord[] = tokens.map((t) => {
      let status = KIND_STATUS[t.kind] ?? 'correct'
      let statusEvidence: AnalysisWord['statusEvidence'] = 'edit'
      const gop = t.align
      if (useAcoustic && status === 'correct' && gop && wordIsOmission(gop)) {
        status = 'omission'
        statusEvidence = 'acoustic'
      }
      return {
        reference: t.kind === 'insert' || t.kind === 'repeat' ? null : t.display,
        hypothesis: status === 'omission' ? null : (t.hyp ?? t.display),
        status,
        statusEvidence,
        confidence: gop ? wordConfidence(gop) : null,
        startMs: gop ? Math.round(gop.t0 * 1000) : null,
        endMs: gop ? Math.round(gop.t1 * 1000) : null,
        gop,
        level: null,
        needsAttention: null,
      }
    })

    const tally = { omissions: 0, substitutions: 0, insertions: 0, repetitions: 0 }
    for (const w of words) {
      if (w.status === 'omission') tally.omissions += 1
      else if (w.status === 'substitution') tally.substitutions += 1
      else if (w.status === 'insertion') tally.insertions += 1
      else if (w.status === 'repetition') tally.repetitions += 1
    }

    const edits: AnalysisEdit[] = []
    let ri = 0
    let hi = 0
    for (const w of words) {
      if (w.status === 'correct') {
        ri += 1
        hi += 1
        continue
      }
      edits.push({
        type: w.status,
        referenceWord: w.reference,
        hypothesisWord: w.hypothesis,
        referenceIndex: w.reference ? ri : null,
        hypothesisIndex: w.hypothesis ? hi : null,
        confidence: w.confidence,
      })
      if (w.reference) ri += 1
      if (w.hypothesis) hi += 1
    }

    const calibrated = contentFromEdits(
      counts.nRef,
      tally.omissions,
      tally.substitutions,
      tally.insertions,
    )
    // Strict stays on the raw edit alignment: the published rule is defined over
    // what was said versus what was written, with no softening of either side.
    const strict = editsFromTexts(ref, hypothesis, false).content
    const mode: ContentMode = input.contentMode ?? 'calibrated'
    const content: ContentScore = {
      score: mode === 'strict' ? strict : calibrated,
      maxScore: 5,
      mode,
      strict,
      calibrated,
      referenceWords: counts.nRef,
      chargedErrors: tally.omissions + tally.substitutions + tally.insertions,
    }

    const timedPauses = pausesFromWords(gopWords, duration)
    const prosody = prosodyFromAlignment({
      words: gopWords,
      pauses: timedPauses,
      referenceText: ref,
      duration,
      alignOk: gopWords.length > 0,
      acoustic: acousticFeatures(pcm.samples, pcm.sampleRate),
      spokenWords: counts.nRef - tally.omissions,
    })

    const shown = displayPauses(gopWords)
    const pauses: PauseSpan[] = []
    for (const p of timedPauses) {
      if (p.where === 'mid' && !shown.some((g) => g.t0 === p.t0 && g.t1 === p.t1)) continue
      const startMs = Math.round(p.t0 * 1000)
      const endMs = Math.round(p.t1 * 1000)
      pauses.push({
        startMs,
        endMs,
        durMs: endMs - startMs,
        where: p.where,
        afterWordIndex: lastWordEndingBy(words, startMs),
      })
    }

    const backend = await this.scoring
    const backendResult = await backend.score({
      durationMs: pcm.durationMs,
      words,
      gopWords,
      hypothesis,
      reference: ref,
      prosody,
      content,
    })

    // Scored words line up with gopWords; insertions carry no alignment and so
    // no band. A word that was not spoken does not get one either: the aligner
    // still placed it, so the head still returns a band, but grading the
    // pronunciation of something nobody said says nothing.
    if (backendResult?.words?.length) {
      let i = 0
      for (const w of words) {
        if (!w.gop) continue
        const scored = backendResult.words[i++]
        if (!scored) break
        if (w.status === 'omission') continue
        w.level = scored.level
        w.needsAttention = scored.needsAttention
      }
    }

    const scores: Scores = {
      backend: backendResult?.backend ?? backend.name,
      content: content.score,
      pronunciation: backendResult?.pronunciation ?? null,
      fluency: backendResult?.fluency ?? null,
      overall: backendResult?.overall ?? null,
    }

    return {
      version: { api: 'v1', package: PACKAGE_VERSION },
      reference: ref,
      hypothesis,
      durationMs: Math.round(pcm.durationMs),
      words,
      edits,
      analysis: tally,
      content,
      prosody,
      pauses,
      tips: buildTips({ words, prosody, pauses }),
      scores,
    }
  }
}

function lastWordEndingBy(words: AnalysisWord[], startMs: number): number | null {
  let idx: number | null = null
  for (let i = 0; i < words.length; i++) {
    const end = words[i]!.endMs
    if (end != null && end <= startMs) idx = i
  }
  return idx
}
